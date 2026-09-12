# Web export

One-shot build steps. Each reads the research artefacts — BIOM tables, the 416
fold checkpoints, the embedding text files — and writes static files into
`data/web/`, which is what the site serves. Nothing here runs at request time.

## Order

```bash
# 0. Inputs, already in the repository or beside it:
#      script/social_niche_embedding_100.txt
#      script/phylo_embed_PCA_100.txt
#      script/taxmap_slv_ssu_ref_nr_138.2.txt
#      script/trait_predcit.csv, script/bacDive.csv, script/agg_bac.csv
#      data/otu_seq/feces_seq_16S_silva.fasta
#      data/healthy_disease_predict/model/<fold>/members/*/model.pth
#      ../microbial-embeddings/analysis/Disease_classification_loo/Data/loo_all_diseases/data/

# 1. Atlas arrays. The notebook is the reference implementation; this build
#    only reads what it wrote.
jupyter nbconvert --execute script/atlas_export.ipynb

# 2. Classifier, vocabulary, reference scores, examples.
.venv-export/bin/python script/web_export/export_dysbiosis.py

# 3. Trait probabilities and BacDive overrides.
.venv-export/bin/python script/web_export/export_traits.py

# 4. Quantised similarities, download formats, vsearch database, manifest.
#    Last, because it rewrites meta.json.
.venv-export/bin/python script/web_export/export_assets.py \
    --site-url https://your.domain

# 5. The fixture the browser regression test compares against.
.venv-export/bin/python script/web_export/export_golden.py
```

The dysbiosis export takes about 35 minutes, almost all of it scoring the
10,276 reference samples through 13 folds twice — once as the ensemble the site
runs, once as each sample's own held-out fold. That second pass is what makes
the AUC on the site the honest one.

## What each script writes

| Script | Files |
|---|---|
| `export_dysbiosis.py` | `vocab.json`, `dysbiosis_encoder.onnx`, `dysbiosis_embed.f16.bin`, `ref_scores.json`, `metrics.json`, `examples/`, `examples.json` |
| `export_traits.py` | `traits_proba.f16.bin`, `traits.json`, `bacdive.json` |
| `export_assets.py` | `nbr_*_sim.u8.bin`, `sne.f16.bin`, `download/*`, `manifest.json`, and the rewrite of `meta.json` |
| `export_golden.py` | `tests/fixtures/golden.json` |
| `export_assets.py` | `data/server/otu_refseqs.fasta` (outside the web root) |

## Two things that are easy to get wrong

**The vocabulary.** `otu_attention.Fid` builds its index as
`['<pad>', '<unk>'] + sorted(feature_ids)`. Nothing stores that order because
nothing needs to — it is derived. The ids come from the BIOM tables and the
order is `sorted`, so a vocabulary rebuilt from the wrong table, or from a
table with an extra feature, silently shifts every index and every score. The
export asserts the size against the checkpoint's embedding shape before it
writes anything.

**Which rows carry information.** All 416 checkpoints share one frozen
embedding table, and in it 5,171 of the 14,021 rows are exactly zero. Those are
the OTUs the training tables contain but the pretrained embedding file does
not, plus the two markers. Only 8,850 OTUs are real, and those are the only
ones with sequences in `feces_seq_16S_silva.fasta`, which is why the vsearch
database holds 8,850 records rather than 14,093.

## Not in this directory

`atlas_export.ipynb` and `traits_predict.ipynb` remain the reference
implementations for the atlas arrays and the trait table. These scripts read
their outputs; `export_traits.py` refits the same forests with the same seed
only because the confidence band needs probabilities the notebook deliberately
leaves empty.
