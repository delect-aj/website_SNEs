"""Write the fixture the browser-side regression test compares against.

The design document's first verification item: take real samples, run the
Python pipeline, run the browser's path, and assert the two agree. Replicating
the preprocessing wrongly makes the whole site wrong, and it would be wrong
quietly — the numbers would still look like numbers.

The fixture deliberately records the *inputs* (raw counts) and the *outputs*
(sequence, abundance, mask) rather than only a score. When the test fails, the
first question is which step drifted, and a logit difference cannot answer it.

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.path.join(
        os.path.dirname(REPO), "microbial-embeddings", "analysis",
        "Disease_classification_loo", "Data", "loo_all_diseases", "data"))
    parser.add_argument("--web", default=os.path.join(REPO, "data", "web"))
    parser.add_argument("--out", default=os.path.join(REPO, "tests", "fixtures",
                                                      "golden.json"))
    args = parser.parse_args()

    with open(os.path.join(args.web, "vocab.json")) as handle:
        vocabulary = json.load(handle)
    vocab = ["<pad>", "<unk>"] + vocabulary["ids"]
    vocab_index = {otu: i for i, otu in enumerate(vocab)}
    num_steps = vocabulary["num_steps"]
    d_model = vocabulary["d_model"]

    table_embedding = np.fromfile(os.path.join(args.web,
                                               "dysbiosis_embed.f16.bin"),
                                  dtype=np.float16).astype(np.float32)
    table_embedding = table_embedding.reshape(vocabulary["n_tokens"], d_model)

    import onnxruntime as ort
    session = ort.InferenceSession(os.path.join(args.web,
                                                "dysbiosis_encoder.onnx"),
                                   providers=["CPUExecutionProvider"])

    fixture = {
        "num_steps": num_steps,
        "d_model": d_model,
        "samples": [],
    }

    for disease in DEFAULT_FOLDS:
        path = os.path.join(args.data_root, disease, "test_loo.biom")
        if not os.path.exists(path):
            continue
        feature_ids, sample_ids, counts = read_biom(path)
        dense_all = counts.toarray().T
        indices = np.array([vocab_index[f] for f in feature_ids], dtype=np.int64)

        # Spread the picks across the fold rather than taking the first few,
        # which can all come from one study.
        step = max(1, dense_all.shape[0] // SAMPLES_PER_FOLD)
        picked = list(range(0, dense_all.shape[0], step))[:SAMPLES_PER_FOLD]

        for row in picked:
            dense = dense_all[row]
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
            fixture["samples"].append({
                "name": f"{disease}/{sample_ids[row]}",
                "disease": disease,
                "sample_id": str(sample_ids[row]),
                "n_otus": int(nonzero.size),
                # Raw counts, so the test runs the browser's own ranking over
                # the same numbers Python ranked.
                "counts": {str(feature_ids[i]): float(dense[i]) for i in nonzero},
                "expected": {
                    "features": [int(v) for v in features[0]],
                    "abundance": [round(float(v), 7) for v in abundance[0]],
                    "mask": [int(v) for v in mask[0]],
                    "logit": float(logit[0][0]),
                },
            })

    # The feature ids are the model's own vocabulary; the browser builds its
    # dense space from whatever the sample carries, so the test passes these
    # in explicitly.
    fixture["feature_ids"] = vocab[2:]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(fixture, handle, separators=(",", ":"))

    sizes = [s["n_otus"] for s in fixture["samples"]]
    print(f"{len(fixture['samples'])} samples written to {args.out}")
    print(f"  taxa per sample: min {min(sizes)}, median "
          f"{int(np.median(sizes))}, max {max(sizes)}")
    print(f"  over {num_steps}: {sum(1 for s in sizes if s > num_steps)}")
    print(f"  file size: {os.path.getsize(args.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
