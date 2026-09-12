"""Export the trait layer the atlas card and its confidence band need.

The card shows one cross-validated AUC per trait, and the band shows where a
query OTU falls among labelled members of each class *on the same probability
axis*. That second requirement is why this script has to refit the forests.

`traits_predict.ipynb` deliberately leaves the probability empty on rows that
carry a Traitar label: those OTUs were in the training set, so their
in-sample probability is close to 1 and says nothing. The band has to place
exactly those rows, so their out-of-sample probability is not the right
quantity either -- what it needs is one consistent number per OTU, produced by
one model, so that both classes are measured the same way. The forest is
therefore refitted here with the notebook's own hyperparameters and seed, and
its probabilities over every labelled OTU are written out.

Outputs, all into `data/web/`:

    traits_proba.f16.bin   float16 probabilities over labelled OTUs
    traits.json            trait order, offsets, class order, AUCs, counts
    bacdive.json           curated measurements, which override predictions

Run:

    .venv-export/bin/python script/web_export/export_traits.py
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_export.trait_labels import (METABOLIC, TRAITS, TRAIT_LABELS,
                                     VALUE_LABELS, load_bacdive, load_embedding,
                                     load_taxonomy, load_traitor)  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The bar has three confidence steps and the weakest is hidden by default.
# 0.65 is not a decoration: it is the line below which the paper's own figure
# stops being informative, and hiding those traits is the only thing keeping a
# 0.55 prediction from looking like a 0.90 one.
AUC_TRUSTED = 0.80
AUC_HIDDEN_BELOW = 0.65


def forest():
    """A fresh classifier per fit, matching `traits_predict.ipynb` exactly."""
    return RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1,
                                  class_weight="balanced")


def trait_level_auc(auc_csv):
    """Mean leave-one-phylum-out AUC per trait, from the notebook's step 1."""
    table = pd.read_csv(auc_csv)
    return table.groupby("traits_type")["auc"].mean().to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-dir", default=os.path.join(REPO, "script"))
    parser.add_argument("--auc-csv", default=os.path.join(
        REPO, "data", "traits_predict", "auc_res.csv"))
    parser.add_argument("--trait-table", default=os.path.join(
        REPO, "data", "traits_predict", "trait_table.csv"))
    parser.add_argument("--out", default=os.path.join(REPO, "data", "web"))
    args = parser.parse_args()

    embedding = load_embedding(os.path.join(args.script_dir,
                                            "social_niche_embedding_100.txt"))
    taxonomy = load_taxonomy(os.path.join(args.script_dir,
                                          "taxmap_slv_ssu_ref_nr_138.2.txt"))
    traitor = load_traitor(os.path.join(args.script_dir, "trait_predcit.csv"))
    bacdive = load_bacdive(os.path.join(args.script_dir, "bacDive.csv"),
                           os.path.join(args.script_dir, "agg_bac.csv"),
                           embedding.index)
    print(f"{len(embedding)} OTUs, {len(traitor)} with Traitar labels, "
          f"{len(bacdive)} with BacDive records")

    # The trait table was written by the notebook; its row order is the
    # embedding's, and the card indexes everything by that order.
    table = pd.read_csv(args.trait_table, index_col="otu_id")
    if list(table.index) != list(embedding.index):
        raise ValueError("trait_table.csv and the embedding file disagree on "
                         "OTU order; the site indexes both positionally")

    auc_by_trait = trait_level_auc(args.auc_csv)

    blocks = []
    meta_traits = {}
    offset = 0
    for trait in TRAITS:
        labelled = traitor[trait].dropna()
        expected = (table[f"{trait}_source"] == "Traitar").sum()
        if len(labelled) != expected:
            raise AssertionError(
                f"{trait}: {len(labelled)} Traitar labels here but "
                f"{expected} rows carry that source in the trait table")

        model = forest().fit(embedding.loc[labelled.index], labelled.values)
        classes = [str(c) for c in model.classes_]
        probabilities = model.predict_proba(embedding)

        labelled_mask = embedding.index.isin(labelled.index)
        block = probabilities[labelled_mask].astype(np.float16)

        auc = float(auc_by_trait.get(trait, float("nan")))
        groups = {str(name): int((labelled.values == name).sum())
                  for name in model.classes_}
        meta_traits[trait] = {
            "label": TRAIT_LABELS[trait],
            "auc": round(auc, 4),
            "displayed": bool(auc >= AUC_HIDDEN_BELOW),
            "classes": classes,
            "value_labels": {str(v): VALUE_LABELS.get(v, str(v))
                             for v in model.classes_},
            "group_sizes": groups,
            "measured": int(len(labelled)),
            "offset": offset,
            "count": int(block.shape[0]),
            "n_classes": int(block.shape[1]),
        }
        blocks.append(block)
        offset += block.shape[0]
        shown = "shown" if meta_traits[trait]["displayed"] else "hidden"
        print(f"  {trait:20s} AUC {auc:.3f}  {len(labelled):5d} labelled  "
              f"{classes}  {shown}")

    # Traits do not share a class count, so the blocks are written flat and
    # `traits.json` records each one's offset, row count and width.
    np.concatenate([block.ravel() for block in blocks]).tofile(
        os.path.join(args.out, "traits_proba.f16.bin"))

    # BacDive measurements, keyed by OTU id. Only the traits the card shows
    # are carried, and only where the record is a definite value.
    measurements = {}
    for otu, row in bacdive.iterrows():
        values = {}
        for trait in TRAITS:
            value = row[trait]
            if pd.isna(value):
                continue
            # The sugar columns arrive as 1.0/0.0 floats; the labels elsewhere
            # in the site are "1"/"0", so normalise before they are compared.
            if isinstance(value, float) and float(value).is_integer():
                value = int(value)
            values[trait] = str(value)
        if values:
            measurements[otu] = values

    with open(os.path.join(args.out, "traits.json"), "w") as handle:
        json.dump({
            "order": list(TRAITS),
            "traits": meta_traits,
            "bacdive_otus": len(measurements),
            "thresholds": {"trusted": AUC_TRUSTED,
                           "hidden_below": AUC_HIDDEN_BELOW},
        }, handle, separators=(",", ":"))

    with open(os.path.join(args.out, "bacdive.json"), "w") as handle:
        json.dump(measurements, handle, separators=(",", ":"))

    hidden = [t for t in TRAITS if not meta_traits[t]["displayed"]]
    print(f"{len(TRAITS) - len(hidden)} traits displayed, {len(hidden)} hidden "
          f"below AUC {AUC_HIDDEN_BELOW}: {hidden}")
    print(f"{len(measurements)} OTUs carry a BacDive measurement")
    for name, size in [("traits_proba.f16.bin", None), ("traits.json", None),
                       ("bacdive.json", None)]:
        path = os.path.join(args.out, name)
        print(f"  {name:24s} {os.path.getsize(path) / 1e3:8.1f} kB")


if __name__ == "__main__":
    main()
