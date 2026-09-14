"""Write taxonomy.json: the SILVA lineage of every OTU in the model's vocabulary.

The dysbiosis result lists the taxa the model attended to, by id and lineage.
The atlas's otus.json cannot serve that: it is 16 MB, and it covers only the
8,850 vocabulary OTUs that carry an embedding, while attention can land on any
of the 14,019. This file holds just the lineage, one string per OTU:

    {"AB002518.1.1416": "Bacteria;Bacillota;...;Streptococcus;Streptococcus hyointestinalis"}

The ranks are the first six fields of the SILVA path, then the species from
organism_name by the same rule the atlas uses (atlas_export.ipynb), so a taxon
reads the same on both pages. OTUs absent from the SILVA 138.2 taxonomy map are
left out.

Standard library only; refreshes manifest.json:

    python3 script/web_export/export_taxonomy.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_examples import refresh_manifest  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB = os.path.join(REPO, "data", "web")
TAXMAP = os.path.join(REPO, "script", "taxmap_slv_ssu_ref_nr_138.2.txt")

# Copied from species_name() in atlas_export.ipynb; the two must agree.
NOT_A_SPECIES = {"uncultured", "unidentified", "unclassified", "metagenome",
                 "bacterium", "organism", "sp.", "rumen", "gut", "human"}


def species_name(organism, genus):
    words = organism.replace("[", "").replace("]", "").split()
    if len(words) < 2 or not genus:
        return ""
    first, second = words[0], words[1]
    if (first.lower() in NOT_A_SPECIES or second.lower() in NOT_A_SPECIES
            or not second.isalpha() or not second.islower()
            or first not in genus.replace("[", "").replace("]", "")):
        return ""
    return f"{first} {second}"


def main():
    with open(os.path.join(WEB, "vocab.json")) as handle:
        wanted = set(json.load(handle)["ids"])

    lineages = {}
    with open(TAXMAP) as handle:
        next(handle)
        for line in handle:
            accession, start, stop, path, organism, _ = line.rstrip("\n").split("\t")
            otu = f"{accession}.{start}.{stop}"
            if otu not in wanted:
                continue
            ranks = (path.split(";") + [""] * 6)[:6]
            ranks.append(species_name(organism, ranks[5]))
            lineages[otu] = ";".join(ranks).rstrip(";")

    with open(os.path.join(WEB, "taxonomy.json"), "w") as handle:
        json.dump(lineages, handle, separators=(",", ":"))
    refresh_manifest(WEB)
    print(f"  {len(lineages)} of {len(wanted)} vocabulary OTUs have a SILVA lineage")


if __name__ == "__main__":
    main()
