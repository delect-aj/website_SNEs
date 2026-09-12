"""Trait label loading, mirroring `script/traits_predict.ipynb`.

The notebook stays the reference implementation; this module is the same
loading code factored out so the website export can reuse it. The export
asserts the counts printed in the notebook, so a divergence between the two
surfaces as a failure rather than as a quietly different trait table.

The two label sources are deliberately different things:

Traitar
    Run on the ~1,100 OTUs that map to a representative genome (Methods
    L424-426). These are the labels the forest is trained and scored against.
BacDive
    Curated wet-lab measurements, joined through the SILVA accession
    (Methods L426-430). These are held out from training and are what the card
    shows as measured rather than inferred.
"""

import numpy as np
import pandas as pd

BIG4 = ["Bacillota", "Bacteroidota", "Actinomycetota", "Pseudomonadota"]
BIG3 = BIG4[:3]

METABOLIC = ["Lactose", "Salicin", "Glycerol", "Melibiose",
             "Maltose", "Sucrose", "Trehalose", "Sorbitol"]

TRAITS = {
    "Oxygen_Preference": (BIG4, 4, 3),
    "Gram_Status": (BIG3, 6, 2),
    "Motility": (BIG3, 6, 2),
    "Spore_Formation": (BIG4, 6, 2),
    **{m: (BIG3, 4, 2) for m in METABOLIC},
}

# What each trait's row is called on the site, and which values it can take.
TRAIT_LABELS = {
    "Oxygen_Preference": "Oxygen preference",
    "Gram_Status": "Gram stain",
    "Motility": "Motility",
    "Spore_Formation": "Spore formation",
    "Lactose": "Lactose",
    "Salicin": "Salicin",
    "Glycerol": "Glycerol",
    "Melibiose": "Melibiose",
    "Maltose": "Maltose",
    "Sucrose": "Sucrose",
    "Trehalose": "Trehalose",
    "Sorbitol": "Sorbitol",
}

VALUE_LABELS = {
    "aerobic": "aerobic", "anaerobic": "anaerobic",
    "facultatively": "facultatively anaerobic",
    "positive": "positive", "negative": "negative",
    1: "yes", 0: "no",
}

# Which class the probability axis is drawn on when a value has several. The
# band is always plotted for the class the queried OTU holds, and both groups
# of labelled OTUs are placed on that same axis.
NOTABLE_CLASS = {"Oxygen_Preference": "anaerobic", "Gram_Status": "negative"}


def load_embedding(path):
    """Read the GloVe-format SNE table, dropping the ``<unk>`` marker row."""
    embedding = pd.read_csv(path, header=None, sep=" ",
                            low_memory=False, index_col=0)
    return embedding.drop(index="<unk>", errors="ignore")


def load_taxonomy(path):
    """SILVA ranks indexed by ``<accession>.<start>.<stop>``, i.e. by OTU id."""
    tax = pd.read_csv(path, sep="\t", low_memory=False)
    ranks = tax["path"].str.split(";", expand=True).iloc[:, :7]
    ranks.columns = ["k", "p", "c", "o", "f", "g", "s"]
    ranks.index = (tax.iloc[:, 0].astype(str) + "."
                   + tax.iloc[:, 1].astype(str) + "."
                   + tax.iloc[:, 2].astype(str)).values
    return ranks


def combine_labels(frame, mapping, exclusive=False):
    """Collapse indicator columns into one categorical column.

    `mapping` is ``{column: label}``, applied in order, first match wins.
    ``exclusive=True`` requires exactly one indicator, as the three oxygen
    classes do -- an organism that claims two of them has no single label.
    """
    out = pd.Series(np.nan, index=frame.index, dtype=object)
    for column, label in mapping.items():
        out[out.isna() & (frame[column] == 1)] = label
    if exclusive:
        out[frame[list(mapping)].sum(axis=1) != 1] = np.nan
    return out


def load_traitor(path):
    """Traitar genome predictions: the labels the forest is fitted on."""
    traitor = (pd.read_csv(path, index_col=0).astype(int).replace(3, 1)
               .rename(columns={"D-Sorbitol": "Sorbitol"}))
    out = pd.DataFrame({
        "Oxygen_Preference": combine_labels(
            traitor, {"Aerobe": "aerobic", "Facultative": "facultatively",
                      "Anaerobe": "anaerobic"}, exclusive=True),
        "Gram_Status": combine_labels(
            traitor, {"Gram negative": "negative", "Gram positive": "positive"},
            exclusive=True),
        "Motility": traitor["Motile"],
        "Spore_Formation": traitor["Spore formation"],
    })
    return out.join(traitor[METABOLIC])


def load_bacdive(bacdive_csv, agg_csv, index):
    """BacDive measurements re-indexed onto OTU ids.

    BacDive is keyed by bare accession while `index` is
    ``<accession>.<start>.<stop>``, so the join goes through the accession.
    """
    agg = pd.read_csv(agg_csv)
    agg["level_3"] = agg["level_3"].str.capitalize()
    agg = agg[agg["level_3"].isin(METABOLIC)
              & (agg["level_2"] == "builds_acid_from") & (agg["type"] == 1)]
    sugars = dict(zip(agg["terms"], agg["level_3"]))

    table = (pd.read_csv(bacdive_csv, low_memory=False)
             .drop_duplicates(subset="16s_ID")
             .set_index("16s_ID")
             .replace({"NA": np.nan, "": np.nan, "-": "no", "+": "yes",
                       "+;NA": np.nan, "mixed": np.nan, "variable": np.nan,
                       "no;yes": np.nan, "negative;positive": np.nan,
                       "negative;variable": np.nan}))
    out = pd.DataFrame({
        "Oxygen_Preference": combine_labels(
            table, {"aerobe": "aerobic", "facultative.anaerobe": "facultatively",
                    "anaerobe": "anaerobic"}),
        "Gram_Status": table["gram_stain"],
        "Motility": table["motility"],
        "Spore_Formation": table["spore_formation"],
    }).join(table[list(sugars)].rename(columns=sugars)).replace({"yes": 1, "no": 0})

    accession = pd.Index(index).str.split(".").str[0]
    keep = accession.isin(out.index)
    out = out.loc[accession[keep]]
    out.index = pd.Index(index)[keep]
    return out
