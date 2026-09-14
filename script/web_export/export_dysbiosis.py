"""Export everything the /dysbiosis page needs.

What this writes into ``data/web/``:

    vocab.json                  OTU id -> model index, in vocabulary order
    dysbiosis_embed.f16.bin     the frozen embedding table, float16
    dysbiosis_encoder.onnx      the 13-fold ensemble, no embedding inside
    ref_scores.json             reference cohort scores, split by group
    metrics.json                the AUCs the page is allowed to quote
    examples/*.json             one-click example samples

The order of operations matters. The vocab is read from the BIOM tables and
checked against the checkpoint's embedding shape before anything is exported,
because a vocabulary off by one id produces a model that runs and is wrong.
Every fold's stripped encoder is then checked against the original module, and
the exported ONNX against PyTorch, before a single score is written.

Run with an interpreter that has torch, onnx, onnxruntime, h5py and scipy:

    .venv-export/bin/python script/web_export/export_dysbiosis.py
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otu_attention import OtuAttentionEncoder  # noqa: E402
from web_export import ensemble, preprocessing  # noqa: E402
from web_export.biom_io import read_biom, read_metadata  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DISEASES = ["AS", "ASD", "BD", "CAD", "CRC", "GD", "IBD",
            "IBS", "MS", "OB", "PD", "SZ", "T2DM"]

# Fixed by the paper's Methods and confirmed against pred_test.csv, which the
# golden test reproduces to 2e-7.
NUM_STEPS = 600

# The attention tensor is (batch, 1, 600, 600) floats, so the batch size is a
# memory decision, not a speed one: 32 keeps it near 45 MB.
BATCH_SIZE = 32

MODEL_KWARGS = dict(d_model=100, n_layers=1, n_heads=1, p_drop=0,
                    d_ff=200, head_hidden=64, abund_mode="multiply",
                    linear_branch=False)


def build_vocab(feature_ids):
    """Return the ordered token list the model was trained against.

    ``otu_attention.Fid`` sorts the feature ids and puts the two markers first,
    so the vocabulary is fully determined by the set of ids in the training
    table -- no file records the order because it is derived, not stored.
    """
    return ["<pad>", "<unk>"] + sorted(set(feature_ids))


def load_encoders(ckpt_root, vocab, device, verbose=True):
    """Build one encoder per disease fold, each carrying its own weights.

    Within a fold the members differ only by seed. The one with the highest
    validation AUC is taken, which is what a validation split is for.

    Returns
    -------
    nets : list of OtuAttentionEncoder
    chosen : dict
        Fold name -> ``{member, valid_auc}``, for the metrics file.
    """
    nets, chosen = [], {}
    for disease in DISEASES:
        members_dir = os.path.join(ckpt_root, disease, "members")
        best, best_auc = None, -1.0
        for member in sorted(os.listdir(members_dir)):
            result_path = os.path.join(members_dir, member, "result.json")
            if not os.path.exists(result_path):
                continue
            with open(result_path) as handle:
                result = json.load(handle)
            if result["valid_auc"] > best_auc:
                best, best_auc = member, result["valid_auc"]

        state = torch.load(os.path.join(members_dir, best, "model.pth"),
                           map_location="cpu")
        if state["embedding.weight"].shape[0] != len(vocab):
            raise ValueError(
                f"{disease}/{best}: embedding has "
                f"{state['embedding.weight'].shape[0]} rows but the vocabulary "
                f"built from the BIOM tables has {len(vocab)}")

        net = OtuAttentionEncoder(otu_size=len(vocab), **MODEL_KWARGS)
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"{disease}/{best}: checkpoint does not match the "
                             f"architecture; missing={missing} "
                             f"unexpected={unexpected}")
        net.to(device).eval()
        nets.append(net)
        chosen[disease] = {"member": best, "valid_auc": round(best_auc, 4)}
        if verbose:
            print(f"  {disease}: {best}, validation AUC {best_auc:.4f}")
    return nets, chosen


def preprocess_table(path, vocab_index, num_steps=NUM_STEPS):
    """Read a BIOM table and turn it into model inputs.

    Parameters
    ----------
    path : str
        Path to the ``.biom`` file.
    vocab_index : dict
        OTU id -> vocabulary index, from :func:`build_vocab`.
    num_steps : int
        Sequence length.

    Returns
    -------
    features, abundance, mask : numpy.ndarray
        Model inputs, see :func:`preprocessing.truncate_pad`.
    sample_ids : numpy.ndarray
    table : scipy.sparse.csc_matrix
        The raw counts, for the examples.
    """
    feature_ids, sample_ids, table = read_biom(path)
    dense = table.toarray().T                      # samples x features
    ranked = np.stack([preprocessing.rank_normalize(row) for row in dense])
    indices = np.array([vocab_index[f] for f in feature_ids], dtype=np.int64)
    features, abundance, mask = preprocessing.truncate_pad(ranked, indices,
                                                           num_steps)
    return features, abundance, mask, sample_ids, table


def reference_embedding(nets):
    """Pull the embedding table out of a fold, with its identity asserted.

    Every checkpoint holds the same frozen table; if a future export ever
    disagrees, the shared blob would silently be the wrong one for some folds.
    """
    table = nets[0].embedding.weight.data.clone()
    for net in nets[1:]:
        if not torch.equal(table, net.embedding.weight.data):
            raise ValueError("checkpoints disagree on the embedding table; "
                             "the folds do not share one")
    return table


def gather(table, features):
    """Embedding lookup for a batch."""
    return table[features]


def run_ensemble(model, table, features, abundance, mask, batch_size=BATCH_SIZE):
    """Run the fold ensemble over a whole table and return logits."""
    logits = np.empty(features.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            out, _ = model(
                gather(table, torch.as_tensor(features[start:stop])),
                torch.as_tensor(abundance[start:stop], dtype=torch.float32),
                torch.as_tensor(mask[start:stop]))
            logits[start:stop] = out.squeeze(1).numpy()
    return logits


def run_single(net, features, abundance, mask, batch_size=BATCH_SIZE):
    """Run one intact encoder over a whole table and return logits."""
    logits = np.empty(features.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            out, _ = net(
                torch.as_tensor(features[start:stop]),
                torch.as_tensor(abundance[start:stop], dtype=torch.float32),
                torch.as_tensor(mask[start:stop]))
            logits[start:stop] = out.squeeze(1).numpy()
    return logits


def check_against_training_run(ckpt_root, disease, member, sample_ids, logits,
                               tolerance=1e-4):
    """Assert the exported inputs reproduce the scores the training run saved.

    ``pred_test.csv`` was written by the training code from the same BIOM table
    and the same checkpoint, so agreement here means the vocabulary, the rank
    normalization, the top-600 selection, the padding and the mask all match
    the pipeline the paper's numbers came from. A transcription error in any
    one of them would show up as a large difference, not a subtle one.
    """
    path = os.path.join(ckpt_root, disease, "members", member, "pred_test.csv")
    saved = {}
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split(",")
        label_at = header.index("true_label")
        prob_at = header.index("prob")
        for line in handle:
            fields = line.rstrip("\n").split(",")
            saved[fields[0]] = (int(fields[label_at]), float(fields[prob_at]))

    missing = [s for s in sample_ids if s not in saved]
    if missing:
        raise KeyError(f"{path}: {len(missing)} samples are absent, "
                       f"first {missing[:3]}")

    expected = np.array([saved[s][1] for s in sample_ids])
    predicted = 1.0 / (1.0 + np.exp(-logits))
    difference = float(np.abs(predicted - expected).max())
    if difference > tolerance:
        raise AssertionError(
            f"{disease}/{member}: reproduced probabilities differ from "
            f"{path} by {difference:.3e}, so the preprocessing here is not the "
            f"preprocessing the training run used")
    return difference


def export_onnx(model, path, num_steps=NUM_STEPS, d_model=100):
    """Write the ensemble as a single ONNX graph with a dynamic batch axis."""
    model.eval()
    dummy = (torch.zeros(2, num_steps, d_model),
             torch.ones(2, num_steps),
             torch.ones(2, num_steps, dtype=torch.int64))
    torch.onnx.export(
        model, dummy, path,
        input_names=["inputs", "weight", "mask"],
        output_names=["logit", "attention"],
        dynamic_axes={"inputs": {0: "batch"}, "weight": {0: "batch"},
                      "mask": {0: "batch"}, "logit": {0: "batch"},
                      "attention": {0: "batch"}},
        opset_version=16,
        do_constant_folding=True,
    )
    return path


def check_onnx(model, table, path, features, abundance, mask, batch=32):
    """Assert onnxruntime reproduces PyTorch, logits and attention alike."""
    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    stop = min(batch, features.shape[0])

    with torch.no_grad():
        inputs = gather(table, torch.as_tensor(features[:stop]))
        torch_logit, torch_attn = model(
            inputs,
            torch.as_tensor(abundance[:stop], dtype=torch.float32),
            torch.as_tensor(mask[:stop]))

    onnx_logit, onnx_attn = session.run(None, {
        "inputs": inputs.numpy(),
        "weight": abundance[:stop].astype(np.float32),
        "mask": mask[:stop].astype(np.int64),
    })

    logit_diff = float(np.abs(onnx_logit - torch_logit.numpy()).max())
    attn_diff = float(np.abs(onnx_attn - torch_attn.numpy()).max())
    for name, value in (("logit", logit_diff), ("attention", attn_diff)):
        if not value < 1e-4:
            raise AssertionError(
                f"onnxruntime disagrees with PyTorch on {name} by {value:.3e}")
    return logit_diff, attn_diff


def auc(labels, scores):
    """Rank AUC."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


# The one-click examples show the two ends of the scale: in each of these
# diseases, the control the ensemble scores lowest and the case it scores
# highest. They are chosen, not typical, and the page says so. Only samples
# with 100 to 600 taxa qualify: enough to look like a real community, and no
# more than the model reads, so no truncation tie decides what an example holds.
EXAMPLE_DISEASES = ["CRC", "IBD", "T2DM"]
EXAMPLE_MIN_OTUS = 100


def write_examples(out_dir, data_root, vocab_index, metadata, model, table,
                   reference):
    """Write one-click examples: raw counts plus the score they should reach.

    The counts are stored raw rather than pre-normalized so that clicking an
    example exercises the browser's own preprocessing, the same code the file
    uploads go through. The expected logit is there to be checked against.

    ``reference`` is the sorted ensemble score of the whole reference cohort,
    used only to report where each chosen sample falls.
    """
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for disease in EXAMPLE_DISEASES:
        path = os.path.join(data_root, disease, "test_loo.biom")
        feature_ids, _, _ = read_biom(path)
        features, abundance, mask, sample_ids, counts = preprocess_table(
            path, vocab_index)
        logits = run_ensemble(model, table, features, abundance, mask)
        dense = counts.toarray().T
        n_otus = (dense > 0).sum(axis=1)
        groups = np.array([int(float(metadata["group"][s])) for s in sample_ids])
        eligible = (n_otus >= EXAMPLE_MIN_OTUS) & (n_otus <= NUM_STEPS)

        for group in (0, 1):
            pool = np.flatnonzero(eligible & (groups == group))
            if not pool.size:
                raise ValueError(f"{disease}: no group {group} sample with "
                                 f"{EXAMPLE_MIN_OTUS}-{NUM_STEPS} taxa")
            row = (pool[np.argmin(logits[pool])] if group == 0
                   else pool[np.argmax(logits[pool])])
            label = "case" if group else "control"
            percentile = 100 * np.searchsorted(reference, logits[row]) / len(reference)

            non_zero = np.nonzero(dense[row])[0]
            record = {
                "name": f"{disease} {label}",
                "disease": disease,
                "group": label,
                "sample_id": str(sample_ids[row]),
                "n_otus": int(non_zero.size),
                "counts": {feature_ids[i]: int(dense[row][i]) for i in non_zero},
                "expected_logit": round(float(logits[row]), 5),
            }
            name = f"{disease.lower()}_{label}.json"
            with open(os.path.join(out_dir, name), "w") as handle:
                json.dump(record, handle, separators=(",", ":"))
            written.append({"file": name, "label": record["name"],
                            "sample_id": record["sample_id"],
                            "n_otus": record["n_otus"]})
            print(f"  {name}: {record['sample_id']}, {non_zero.size} OTUs, "
                  f"logit {logits[row]:.3f}, reference percentile "
                  f"{percentile:.1f} (of {pool.size} eligible {label}s)")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.path.join(
        os.path.dirname(REPO), "microbial-embeddings", "analysis",
        "Disease_classification_loo", "Data", "loo_all_diseases", "data"))
    parser.add_argument("--ckpt-root",
                        default=os.path.join(REPO, "data",
                                             "healthy_disease_predict", "model"))
    parser.add_argument("--out", default=os.path.join(REPO, "data", "web"))
    parser.add_argument("--skip-reference", action="store_true",
                        help="stop before scoring the reference cohort")
    parser.add_argument("--examples-only", action="store_true",
                        help="rewrite examples/ and examples.json from the "
                             "vocab.json and ref_scores.json already in --out; "
                             "needs only the example diseases' test tables")
    parser.add_argument("--metadata", default=None,
                        help="sample metadata TSV (default: DATA_ROOT/metadata.tsv)")
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these folds; a smoke test, the "
                             "resulting reference cohort is incomplete")
    args = parser.parse_args()

    if args.only:
        global DISEASES
        DISEASES = args.only

    os.makedirs(args.out, exist_ok=True)
    metadata_path = args.metadata or os.path.join(args.data_root, "metadata.tsv")

    if args.examples_only:
        write_examples_only(args, metadata_path)
        return

    # ---- vocabulary ------------------------------------------------------ #
    print("vocabulary")
    reference_path = os.path.join(args.data_root, "AS", "train_loo.biom")
    feature_ids, _, _ = read_biom(reference_path)
    vocab = build_vocab(feature_ids)
    print(f"  {len(feature_ids)} features in the BIOM table -> {len(vocab)} "
          f"tokens including the two markers")

    # Every fold must use the same table; the ensemble would be meaningless if
    # one fold read a different index space.
    for disease in DISEASES:
        for part in ("train", "test"):
            path = os.path.join(args.data_root, disease, f"{part}_loo.biom")
            ids, _, _ = read_biom(path)
            if set(ids) != set(feature_ids):
                raise ValueError(
                    f"{path}: feature set differs from {reference_path}")

    vocab_index = {otu: i for i, otu in enumerate(vocab)}
    with open(os.path.join(args.out, "vocab.json"), "w") as handle:
        json.dump({
            "n_tokens": len(vocab),
            "d_model": MODEL_KWARGS["d_model"],
            "pad_index": preprocessing.PAD_INDEX,
            "unk_index": preprocessing.UNK_INDEX,
            "num_steps": NUM_STEPS,
            "ids": vocab[2:],
        }, handle, separators=(",", ":"))

    # ---- encoders -------------------------------------------------------- #
    print("loading one member per fold (highest validation AUC)")
    nets, chosen = load_encoders(args.ckpt_root, vocab, torch.device("cpu"))

    table = reference_embedding(nets)
    # Rows 0 and 1 are the two markers and are zero by construction, so every
    # non-zero row is an OTU that carries information.
    usable = int((table.abs().sum(dim=1) > 0).sum())
    print(f"  shared embedding {tuple(table.shape)}, {usable} non-zero rows "
          f"of {len(vocab) - 2} OTUs")

    # ---- equivalence, before anything is stripped ------------------------ #
    print("checking the embedding substitution")
    features, abundance, mask, sample_ids, _ = preprocess_table(
        os.path.join(args.data_root, "IBD", "test_loo.biom"), vocab_index)
    probe_index = torch.as_tensor(features[:8])
    probe_weight = torch.as_tensor(abundance[:8], dtype=torch.float32)
    probe_mask = torch.as_tensor(mask[:8])
    worst = max(ensemble.check_equivalence(net, probe_index, probe_weight,
                                           probe_mask)
                for net in nets)
    print(f"  max logit difference {worst:.3e}")

    for net in nets:
        ensemble.strip_embedding(net)
    model = ensemble.FoldEnsemble(nets).eval()

    # ---- ONNX ------------------------------------------------------------ #
    onnx_path = os.path.join(args.out, "dysbiosis_encoder.onnx")
    print("exporting ONNX")
    export_onnx(model, onnx_path)
    logit_diff, attn_diff = check_onnx(model, table, onnx_path,
                                       features, abundance, mask)
    print(f"  {os.path.getsize(onnx_path) / 1e6:.2f} MB | "
          f"logit diff {logit_diff:.2e}, attention diff {attn_diff:.2e}")

    fp16 = table.numpy().astype(np.float16)
    fp16.tofile(os.path.join(args.out, "dysbiosis_embed.f16.bin"))
    quantisation = float(np.abs(fp16.astype(np.float32)
                                - table.numpy()).max())
    print(f"  embedding float16 max abs error {quantisation:.2e}")

    if args.skip_reference:
        print("stopping before the reference cohort")
        return

    # ---- reference cohort ------------------------------------------------ #
    # Both numbers below come from the same pass over the 13 test tables.
    #
    # The leave-one-disease-out score is each sample's own fold: the one model
    # that never saw its disease. That is the paper's generalisation estimate
    # and the only performance figure the site is allowed to quote.
    #
    # The ensemble score is the average of all 13 folds, which is what the
    # browser runs. For a sample of disease D, twelve of those folds trained on
    # D, so the ensemble separates this cohort far better (pooled AUC 0.80 vs
    # 0.64) than it would separate a disease from outside it. That number is
    # descriptive of the reference distribution and must not be presented as
    # accuracy.
    print("scoring the reference cohort, one fold at a time")
    metadata = read_metadata(metadata_path, ["group", "disease_name_ab"])

    intact_nets, _ = load_encoders(args.ckpt_root, vocab, torch.device("cpu"),
                                   verbose=False)

    scores, lodo_scores, labels, diseases = [], [], [], []
    seen = set()
    for index, disease in enumerate(DISEASES):
        path = os.path.join(args.data_root, disease, "test_loo.biom")
        started = time.time()
        feat, abun, msk, ids, _ = preprocess_table(path, vocab_index)
        duplicate = seen.intersection(ids)
        if duplicate:
            raise ValueError(f"{path}: samples already scored by another fold: "
                             f"{sorted(duplicate)[:3]}")
        seen.update(ids)

        missing = [s for s in ids if s not in metadata["group"]]
        if missing:
            raise KeyError(f"{path}: {len(missing)} samples absent from "
                           f"metadata.tsv, first {missing[:3]}")

        logits = run_ensemble(model, table, feat, abun, msk)
        lodo_logits = run_single(intact_nets[index], feat, abun, msk)
        check_against_training_run(args.ckpt_root, disease,
                                   chosen[disease]["member"], ids, lodo_logits)

        fold_labels = [int(float(metadata["group"][s])) for s in ids]
        scores.append(logits)
        lodo_scores.append(lodo_logits)
        labels.append(fold_labels)
        diseases.extend([disease] * len(ids))
        print(f"  {disease}: {len(ids)} samples, held-out AUC "
              f"{auc(fold_labels, lodo_logits):.4f}, ensembled "
              f"{auc(fold_labels, logits):.4f} ({time.time() - started:.1f}s)")

    scores = np.concatenate(scores)
    lodo_scores = np.concatenate(lodo_scores)
    labels = np.concatenate(labels)
    diseases = np.array(diseases)

    controls = np.sort(scores[labels == 0])
    cases = np.sort(scores[labels == 1])
    with open(os.path.join(args.out, "ref_scores.json"), "w") as handle:
        json.dump({
            "n_controls": int(len(controls)),
            "n_cases": int(len(cases)),
            "controls": [round(float(v), 4) for v in controls],
            "cases": [round(float(v), 4) for v in cases],
        }, handle, separators=(",", ":"))

    lodo_auc = auc(labels, lodo_scores)
    per_disease = {}
    for disease in DISEASES:
        picked = diseases == disease
        per_disease[disease] = round(auc(labels[picked], lodo_scores[picked]), 4)
    print(f"  leave-one-disease-out AUC: {lodo_auc:.4f}")
    print(f"  ensemble on its own cohort: {auc(labels, scores):.4f} (descriptive, "
          f"not generalisation)")

    with open(os.path.join(args.out, "metrics.json"), "w") as handle:
        json.dump({
            "lodo_auc": round(float(lodo_auc), 4),
            "per_disease_auc": per_disease,
            "reference_auc": round(float(auc(labels, scores)), 4),
            "reference_auc_note": (
                "How well the deployed ensemble separates the cohort it was "
                "trained on. Twelve of the thirteen folds saw each sample's "
                "disease, so this is not a generalisation estimate and is not "
                "shown to visitors; lodo_auc is."),
            "n_reference": int(len(scores)),
            "n_controls": int(len(controls)),
            "n_cases": int(len(cases)),
            "folds": chosen,
            "num_steps": NUM_STEPS,
            "n_tokens": len(vocab),
            "n_informative_otus": usable,
        }, handle, indent=1)

    # ---- examples -------------------------------------------------------- #
    print("writing example samples")
    examples = write_examples(os.path.join(args.out, "examples"), args.data_root,
                              vocab_index, metadata, model, table, np.sort(scores))
    with open(os.path.join(args.out, "examples.json"), "w") as handle:
        json.dump(examples, handle, indent=1)

    print("done")


def write_examples_only(args, metadata_path):
    """Rebuild the examples against the vocabulary and cohort already exported."""
    with open(os.path.join(args.out, "vocab.json")) as handle:
        vocab = ["<pad>", "<unk>"] + json.load(handle)["ids"]
    vocab_index = {otu: i for i, otu in enumerate(vocab)}

    print("loading one member per fold (highest validation AUC)")
    nets, _ = load_encoders(args.ckpt_root, vocab, torch.device("cpu"))
    table = reference_embedding(nets)
    for net in nets:
        ensemble.strip_embedding(net)
    model = ensemble.FoldEnsemble(nets).eval()

    with open(os.path.join(args.out, "ref_scores.json")) as handle:
        cohort = json.load(handle)
    reference = np.sort(np.array(cohort["controls"] + cohort["cases"]))
    metadata = read_metadata(metadata_path, ["group", "disease_name_ab"])

    print("writing example samples")
    examples = write_examples(os.path.join(args.out, "examples"), args.data_root,
                              vocab_index, metadata, model, table, reference)
    with open(os.path.join(args.out, "examples.json"), "w") as handle:
        json.dump(examples, handle, indent=1)


if __name__ == "__main__":
    main()
