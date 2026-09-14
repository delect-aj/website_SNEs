"""Write the fixture the browser-side regression test compares against.

The design document's first verification item: take real samples, run the
Python pipeline, run the browser's path, and assert the two agree. Replicating
the preprocessing wrongly makes the whole site wrong, and it would be wrong
quietly — the numbers would still look like numbers.

The fixture deliberately records the *inputs* (raw counts) and the *outputs*
(sequence, abundance, mask) rather than only a score. When the test fails, the
first question is which step drifted, and a logit difference cannot answer it.

Two sets of samples go in. The reference-cohort picks come from the BIOM
tables and cover a spread of studies, a three-class trait and one sample deeper
than the sequence length. The one-click examples come from `data/web/`, which
is what makes the test cover the samples a visitor can actually click — those
carry an `expected_logit` that nothing else checks.

Both sides of the test read the same float16 embedding blob, so quantisation
is not a difference between them; it is measured separately by the export.

Run after `export_dysbiosis.py`:

    .venv-export/bin/python script/web_export/export_golden.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_export import preprocessing  # noqa: E402
from web_export.biom_io import read_biom  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enough folds to cover a three-class trait, a sample that overflows the
# sequence length, and the ordinary case, without making the fixture large.
DEFAULT_FOLDS = ["AS", "ASD", "CAD", "CRC", "IBD", "IBS", "MS", "PD", "SZ", "T2DM"]
SAMPLES_PER_FOLD = 3


def load_inputs(web_dir):
    """The vocabulary, the frozen embedding table and the exported graph.

    Everything the two sample builders need, loaded once.
    """
    with open(os.path.join(web_dir, "vocab.json")) as handle:
        vocabulary = json.load(handle)
    vocab = ["<pad>", "<unk>"] + vocabulary["ids"]
    vocab_index = {otu: i for i, otu in enumerate(vocab)}

    table_embedding = np.fromfile(os.path.join(web_dir,
                                               "dysbiosis_embed.f16.bin"),
                                  dtype=np.float16).astype(np.float32)
    table_embedding = table_embedding.reshape(vocabulary["n_tokens"],
                                              vocabulary["d_model"])

    import onnxruntime as ort
    session = ort.InferenceSession(os.path.join(web_dir,
                                                "dysbiosis_encoder.onnx"),
                                   providers=["CPUExecutionProvider"])
    return vocabulary, vocab_index, session, table_embedding


def expected_arrays(session, table_embedding, dense, indices, num_steps):
    """Run the Python pipeline over one sample.

    Parameters
    ----------
    dense : numpy.ndarray
        Raw counts of one sample, one entry per column of the caller's index
        space.
    indices : numpy.ndarray
        Vocabulary index of each column of `dense`.

    Returns
    -------
    tuple
        ``(expected, n_otus)`` — the arrays and logit the test compares, and
        the number of non-zero taxa, which decides how the test compares them.
    """
    ranked = preprocessing.rank_normalize(dense)
    features, abundance, mask = preprocessing.truncate_pad(
        ranked[None, :], indices, num_steps)

    gathered = table_embedding[features[0]].astype(np.float32)
    logit, _ = session.run(None, {
        "inputs": gathered[None, :, :],
        "weight": abundance.astype(np.float32),
        "mask": mask.astype(np.int64),
    })

    nonzero = np.nonzero(dense)[0]
    return {
        "features": [int(v) for v in features[0]],
        "abundance": [round(float(v), 7) for v in abundance[0]],
        "mask": [int(v) for v in mask[0]],
        "logit": float(logit[0][0]),
    }, int(nonzero.size)


def fold_samples(data_root, vocab_index, session, table_embedding, num_steps,
                 folds=DEFAULT_FOLDS, per_fold=SAMPLES_PER_FOLD):
    """Reference-cohort samples, one table per fold."""
    samples = []
    for disease in folds:
        path = os.path.join(data_root, disease, "test_loo.biom")
        if not os.path.exists(path):
            continue
        feature_ids, sample_ids, counts = read_biom(path)
        dense_all = counts.toarray().T
        indices = np.array([vocab_index[f] for f in feature_ids], dtype=np.int64)

        # Spread the picks across the fold rather than taking the first few,
        # which can all come from one study.
        step = max(1, dense_all.shape[0] // per_fold)
        picked = list(range(0, dense_all.shape[0], step))[:per_fold]

        for row in picked:
            dense = dense_all[row]
            expected, n_otus = expected_arrays(session, table_embedding, dense,
                                               indices, num_steps)

            nonzero = np.nonzero(dense)[0]
            samples.append({
                "name": f"{disease}/{sample_ids[row]}",
                "disease": disease,
                "sample_id": str(sample_ids[row]),
                "n_otus": n_otus,
                # Raw counts, so the test runs the browser's own ranking over
                # the same numbers Python ranked.
                "counts": {str(feature_ids[i]): float(dense[i])
                           for i in nonzero},
                "expected": expected,
            })
    return samples


def example_samples(web_dir, vocabulary, vocab_index, session,
                    table_embedding):
    """The one-click examples, encoded the way the browser encodes them.

    A visitor's page builds the dense row from the sample's own ids; here it is
    built over the vocabulary's order, which is the index space the BIOM tables
    use as well, so these samples carry the same kind of expected arrays as the
    fold samples and go through the same comparison.

    The counts are listed in the order the example file lists them, which is
    ascending vocabulary index — the order the browser's tie-breaking sees.
    """
    num_steps = vocabulary["num_steps"]
    ids = vocabulary["ids"]
    index_of = {otu: position for position, otu in enumerate(ids)}
    count_vocab = len(ids) + 2
    indices = np.arange(2, count_vocab, dtype=np.int64)

    with open(os.path.join(web_dir, "examples.json")) as handle:
        records = json.load(handle)

    samples = []
    for record in records:
        with open(os.path.join(web_dir, "examples", record["file"])) as handle:
            example = json.load(handle)

        # Every id the page would mask has to be visible here instead: an
        # unmapped id would make this expectation disagree with the browser for
        # a reason the test could not name.
        unknown = [otu for otu in example["counts"] if otu not in index_of]
        if unknown:
            raise KeyError(
                f"{record['file']}: {len(unknown)} ids are not in the "
                f"vocabulary, first {unknown[:3]}")

        dense = np.zeros(count_vocab, dtype=np.float64)
        for otu, count in example["counts"].items():
            dense[index_of[otu]] = count

        expected, n_otus = expected_arrays(session, table_embedding, dense,
                                          indices, num_steps)
        samples.append({
            "name": f"example/{example['sample_id']}",
            "disease": example.get("disease"),
            "sample_id": str(example["sample_id"]),
            "n_otus": n_otus,
            "counts": {otu: float(count)
                       for otu, count in example["counts"].items()},
            "expected": expected,
        })
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.path.join(
        os.path.dirname(REPO), "microbial-embeddings", "analysis",
        "Disease_classification_loo", "Data", "loo_all_diseases", "data"))
    parser.add_argument("--web", default=os.path.join(REPO, "data", "web"))
    parser.add_argument("--out", default=os.path.join(REPO, "tests", "fixtures",
                                                      "golden.json"))
    parser.add_argument("--examples-only", action="store_true",
                        help="keep the fixture's reference-cohort samples and "
                             "rebuild only the examples, after they change")
    parser.add_argument("--skip-examples", action="store_true",
                        help="reference-cohort samples only; the resulting "
                             "fixture is incomplete")
    args = parser.parse_args()

    vocabulary, vocab_index, session, table_embedding = load_inputs(args.web)
    num_steps = vocabulary["num_steps"]
    d_model = vocabulary["d_model"]

    fixture = {
        "num_steps": num_steps,
        "d_model": d_model,
        "samples": [],
    }

    if args.examples_only:
        with open(args.out) as handle:
            kept = json.load(handle)["samples"]
        fixture["samples"].extend(s for s in kept
                                  if not s["name"].startswith("example/"))
    else:
        fixture["samples"].extend(fold_samples(
            args.data_root, vocab_index, session, table_embedding, num_steps))

    if not args.skip_examples:
        fixture["samples"].extend(example_samples(
            args.web, vocabulary, vocab_index, session, table_embedding))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(fixture, handle, separators=(",", ":"))

    sizes = [s["n_otus"] for s in fixture["samples"]]
    examples = sum(1 for s in fixture["samples"]
                   if s["name"].startswith("example/"))
    print(f"{len(fixture['samples'])} samples written to {args.out}")
    print(f"  {examples} of them the one-click examples")
    print(f"  taxa per sample: min {min(sizes)}, median "
          f"{int(np.median(sizes))}, max {max(sizes)}")
    print(f"  over {num_steps}: {sum(1 for s in sizes if s > num_steps)}")
    print(f"  file size: {os.path.getsize(args.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
