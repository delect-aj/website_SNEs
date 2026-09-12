# Microbial social niches — website

The public site for the ecological embeddings of the human gut microbiome. Two
halves: a static site that needs no server process, and one small endpoint that
maps amplicon sequences to reference OTUs.

```
web/                  the site: HTML, CSS, ES modules, vendored dependencies
data/web/             everything the site serves, all built offline
data/server/          the vsearch database, outside the web root
script/               the research notebooks and the export scripts
  web_export/         one-shot build steps; see its README
server/               FastAPI + vsearch, the only running service
tests/js/             the browser path against Python
deploy/               nginx, Docker, systemd, and WEB_CONFIG.md
```

## What is on the site

| Page | What it does |
|---|---|
| `/atlas` | 14,093 gut OTUs on a map, searched by taxon. Each card shows the ecological neighbours and the phylogenetic ones side by side, then the inferred traits with the cross-validated AUC each one earned |
| `/dysbiosis` | Scores one faecal sample against a reference cohort of 10,276, in the visitor's browser. Three inputs: a ready-made example, an OTU table, or rep-seqs plus counts |
| `/download` | The vectors, the model, the trait tables, and a link that opens the TensorFlow Embedding Projector with this data loaded |
| `/cite` | Citation, version, and the numbers you are allowed to quote |

## Running it

```bash
# Serve the site locally. The symlink puts the built data at /data/.
ln -sfn ../data/web web/data
cd web && python3 -m http.server 8080

# The mapping service, only needed for the rep-seqs route.
cd server
OTU_REFSEQS=../data/server/otu_refseqs.fasta \
VSEARCH_BINARY=/path/to/vsearch \
python -m uvicorn app.main:app --port 8000
```

Deployment to a real server is in [`deploy/WEB_CONFIG.md`](deploy/WEB_CONFIG.md).

## Rebuilding the data

`data/web/` is build output. The steps, their order and what each produces are
in [`script/web_export/README.md`](script/web_export/README.md). In short:

```bash
jupyter nbconvert --execute script/atlas_export.ipynb
.venv-export/bin/python script/web_export/export_dysbiosis.py
.venv-export/bin/python script/web_export/export_traits.py
.venv-export/bin/python script/web_export/export_assets.py --site-url https://your.domain
.venv-export/bin/python script/web_export/export_golden.py
```

## Tests

```bash
node tests/js/preprocess.test.mjs          # the browser path against Python
cd server && python -m pytest tests -v     # the endpoint's limits and mapping
```

The first is the one that matters. The site re-implements the paper's
preprocessing in JavaScript, and a mistake there produces a plausible wrong
number rather than an error. See [`tests/js/README.md`](tests/js/README.md) for
what is asserted and the one place the two languages legitimately differ.

## Two numbers worth knowing before reading the code

**The model knows 8,850 OTUs, not 14,093.** The atlas covers all 14,093
embedded OTUs. The classifier's vocabulary is a different 14,019 ids, of which
5,171 have no trained embedding, and only 8,850 have both a vector and a
sequence. Those are the ones `/map` can return and the ones that carry
information.

**The graded AUC is 0.6383, not 0.7976.** The deployed model is an average of
thirteen leave-one-disease-out folds. Twelve of the thirteen saw any given
reference sample's disease during training, so the ensemble separates the
reference cohort with an AUC of 0.80 — a number that says nothing about a
disease it has not seen. The honest estimate is each sample scored by the one
fold that held its disease out: 0.6383, which is what the site quotes and what
`metrics.json` records as `lodo_auc`. `reference_auc` is kept beside it, with a
note saying not to use it.
