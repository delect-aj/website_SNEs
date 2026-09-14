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

---

## Deploying

### What runs where

```
                        visitor's browser
                        ├─ every model inference (onnxruntime-web, WebAssembly)
                        ├─ every preprocessing step
                        └─ every figure
                                 │
  nginx (host) ──────────────────┼──────────────────────────────────────
  ├─ /, /atlas, /dysbiosis, …    static files from site/
  ├─ /data/*                     static files from site/data/
  └─ /map ───────────► map service on 127.0.0.1:8000
                       └─ vsearch against data/otu_refseqs.fasta
```

Only one process runs. The classifier is not on the server: it ships to the
browser as a 4.5 MB ONNX graph, which is what lets a 1-core host serve an
arbitrary number of visitors. The mapping endpoint exists because aligning
sequences is the one thing a browser cannot do.

**Sizing.** 1 vCPU, 2 GB RAM, 20 GB disk. vsearch is single-threaded and the
container is capped at one core; the static site is a few files on disk.

**What a visit costs.** A first visit transfers about 4.8 MB for `/atlas` and
9.2 MB for `/dysbiosis`, gzipped — measured over the files each page fetches.
Most of the second figure is ONNX Runtime itself: 10.9 MB of WebAssembly
uncompressed, 3.0 MB compressed, fetched once and then served from the browser
cache. The server does no work beyond sending files.

### What you need

- The build machine: the one with the BIOM tables and the fold checkpoints.
  See [Rebuilding the data](#rebuilding-the-data).
- The server: Ubuntu 22.04/24.04 or Debian 12 (any distribution with nginx
  ≥ 1.18 works), a domain name pointing at it, and root or sudo.

### Step 1 — build the artefacts

On the build machine, not the server. This produces everything under
`data/web/` plus the vsearch database.

```bash
git clone <this repository> && cd website_SNEs

python -m venv --system-site-packages .venv-export
.venv-export/bin/pip install "onnx==1.15.0" "onnxruntime==1.16.3"

jupyter nbconvert --execute script/atlas_export.ipynb
.venv-export/bin/python script/web_export/export_dysbiosis.py
.venv-export/bin/python script/web_export/export_traits.py
.venv-export/bin/python script/web_export/export_assets.py \
    --site-url https://your.domain
.venv-export/bin/python script/web_export/export_golden.py
```

`export_dysbiosis.py` takes about 35 minutes and verifies itself before writing
anything: it reproduces the training run's `pred_test.csv` for all 10,276
reference samples, then checks onnxruntime against PyTorch. If it fails, do not
deploy the output.

`--site-url` must be the public origin. It goes into the Embedding Projector
config, which Google's server fetches from yours.

The browser dependencies are vendored except for two 10 MB `.wasm` files:

```bash
script/fetch_vendor.sh
```

### Step 2 — copy everything to the server

The layout the configuration expects:

```
/srv/microbial/
├── site/                       <- web/            (nginx document root)
│   └── data/                   <- data/web/
├── data/
│   └── otu_refseqs.fasta       <- data/server/    (deliberately outside site/)
├── server/                     <- server/
└── deploy/                     <- deploy/
```

The reference FASTA is 12 MB that no browser ever requests, so it stays out of
the document root.

```bash
ssh you@server 'sudo mkdir -p /srv/microbial && sudo chown $USER /srv/microbial'
rsync -av --delete web/                    you@server:/srv/microbial/site/
rsync -av --delete data/web/               you@server:/srv/microbial/site/data/
rsync -av --delete data/server/            you@server:/srv/microbial/data/
rsync -av --delete server/ deploy/         you@server:/srv/microbial/
```

Both `--delete` targets are build output, which is what makes re-deploying
idempotent. Never point one at a directory holding anything else.

### Step 3 — run the mapping service

Pick one. Both bind to `127.0.0.1:8000` only; nginx is the single public entry
point, and vsearch behind no rate limit is not something to expose.

#### Docker

```bash
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker "$USER"      # then log out and back in
cd /srv/microbial
docker compose -f deploy/docker-compose.yml up -d --build
```

#### systemd, no Docker

Needs vsearch, which most distributions do not package. A release binary is
the quickest route:

```bash
sudo useradd --system --home /srv/microbial --shell /usr/sbin/nologin microbial
sudo -u microbial python3 -m venv /srv/microbial/venv
sudo -u microbial /srv/microbial/venv/bin/pip install -r /srv/microbial/server/requirements.txt

sudo apt install -y wget
sudo wget -O /tmp/vsearch.tar.gz \
  https://github.com/torognes/vsearch/releases/download/v2.28.1/vsearch-2.28.1-linux-x86_64.tar.gz
sudo tar -xzf /tmp/vsearch.tar.gz -C /tmp
sudo install -m 0755 /tmp/vsearch-2.28.1-linux-x86_64/bin/vsearch /usr/local/bin/vsearch
vsearch --version

sudo install -m 0644 /srv/microbial/deploy/microbial-map.service \
  /etc/systemd/system/microbial-map.service
sudo systemctl daemon-reload
sudo systemctl enable --now microbial-map
```

Either way, confirm the service can see its database before going further:

```bash
curl -s localhost:8000/health
# expect "database_present": true — if not, the path or the bind mount is wrong
# and /map will answer 503
```

### Step 4 — nginx and TLS

Rate-limit zones cannot be declared inside a `server` block, so add one line to
the `http` block of `/etc/nginx/nginx.conf`:

```nginx
limit_req_zone $binary_remote_addr zone=map_limit:10m rate=10r/m;
```

The supplied config has an HTTPS block pointing at a certificate that does not
exist yet, so nginx would refuse to start if you install it now. Two phases,
with the certificate obtained in between:

```bash
sudo apt install -y nginx certbot
sudo mkdir -p /var/www/certbot

# Phase 1: HTTP only, just enough for nginx to start and for certbot to answer
# the ACME challenge. DNS must already resolve to this host.
sudo tee /etc/nginx/sites-available/microbial-embeddings >/dev/null <<'CONF'
server {
    listen 80;
    server_name your.domain;
    root /srv/microbial/site;
    location /.well-known/acme-challenge/ { root /var/www/certbot; }
    location / { return 404; }
}
CONF
sudo ln -sf /etc/nginx/sites-available/microbial-embeddings \
  /etc/nginx/sites-enabled/microbial-embeddings
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

sudo certbot certonly --webroot -w /var/www/certbot -d your.domain

# Phase 2: the real config, which can start because the certificate now exists.
sudo cp /srv/microbial/deploy/nginx.conf \
  /etc/nginx/sites-available/microbial-embeddings
sudo sed -i 's|SITE_ROOT|/srv/microbial/site|' \
  /etc/nginx/sites-available/microbial-embeddings
sudo sed -i 's|microbiome.example.org|your.domain|g' \
  /etc/nginx/sites-available/microbial-embeddings
sudo nginx -t && sudo systemctl reload nginx
```

Certbot installed a renewal timer when it issued the certificate; confirm it
with `systemctl list-timers | grep certbot`. Renewal uses the webroot path
above, which the deployed config keeps serving.

Open the firewall, leaving 8000 closed:

```bash
sudo ufw allow 22,80,443/tcp && sudo ufw enable
```

### Step 5 — verify the deployment

Each check catches a different class of mistake. Run them in order.

```bash
# 1. The service is up and can see its database.
curl -s https://your.domain/health | python3 -m json.tool

# 2. Every file the manifest lists is actually being served.
curl -s https://your.domain/data/manifest.json \
  | python3 -c "import json,sys; [print(f['file']) for f in json.load(sys.stdin)['files']]" \
  | while read -r f; do
      code=$(curl -s -o /dev/null -w '%{http_code}' "https://your.domain/data/$f")
      [ "$code" = 200 ] || echo "MISSING $f ($code)"
    done

# 3. Compression is on. otus.json is 16 MB raw and should arrive as ~1.1 MB.
curl -s -H 'Accept-Encoding: gzip' -o /dev/null -w '%{size_download}\n' \
  https://your.domain/data/otus.json

# 4. The WebAssembly gets the MIME type onnxruntime-web requires. Anything but
#    application/wasm leaves the dysbiosis page dead with no useful error.
curl -sI https://your.domain/assets/vendor/ort/ort-wasm-simd-threaded.wasm \
  | grep -i content-type

# 5. /map round-trips: sequences taken out of the database must come back as
#    themselves, not merely as something.
head -c 40000 /srv/microbial/data/otu_refseqs.fasta > /tmp/probe.fasta
curl -s -F "rep_seqs=@/tmp/probe.fasta" https://your.domain/map | python3 -c "
import json, sys
payload = json.load(sys.stdin)
print(payload['mapped'], 'of', payload['total'], 'mapped')
bad = [(q, s) for q, s in payload['mapping'].items() if q != s]
assert payload['mapped'] == payload['total'] and not bad, bad[:3]
print('every sequence mapped back to itself')"

# 6. Bad input is rejected with 4xx, not 5xx.
printf 'not a fasta\n' > /tmp/bad.txt
curl -s -o /dev/null -w '%{http_code}\n' -F "rep_seqs=@/tmp/bad.txt" \
  https://your.domain/map            # expect 400
```

Then walk the golden path by hand:

1. `/atlas` draws 14,093 points; the colour-by selector changes them; searching
   a genus fills the card beside the map.
2. On a card, "Show the labelled distribution" draws two groups of dots and a
   marker, and a trait with AUC below 0.65 sits folded at the bottom.
3. `/dysbiosis` → *Run an example sample* → a percentile appears with the
   cohort distribution under it, and a contributing taxon links into `/atlas`.
4. The same page with a real rep-seqs FASTA plus counts goes through `/map` and
   produces a score. A mapping rate near zero means the sequences are not SILVA
   138.2 97% OTU reps, so the sample is not comparable to the cohort.

### Re-deploying

```bash
git pull
script/check_vendor.sh           # after a pull: the wasm files are not in git
rsync -av --delete web/          server:/srv/microbial/site/
rsync -av --delete data/web/     server:/srv/microbial/site/data/
# only when the model or the vocabulary changed:
rsync -av --delete data/server/  server:/srv/microbial/data/
```

`ort.min.js` is committed but the two `.wasm` files beside it are not, so a
pull can leave a new loader next to the previous release's wasm — onnxruntime
then fails in the browser without ever mentioning a version.
`script/check_vendor.sh` compares both against the pin and says so.

Browser assets are cached for a month, so after changing anything under
`web/assets/` either rename the file or lower `max-age` in `nginx.conf` — the
old copies will otherwise be served from visitors' caches.

Nothing on the server is stateful. There is no database, no uploads directory
and no session state; `/map` writes a temporary file, runs vsearch and removes
it before answering. A rebuild from scratch is one `rsync`.

---

## Local development

```bash
# The symlink puts the built data at /data/ without copying it.
ln -sfn ../data/web web/data
cd web && python3 -m http.server 8080

# The mapping service, needed only for the rep-seqs route.
cd server
OTU_REFSEQS=../data/server/otu_refseqs.fasta \
VSEARCH_BINARY=/path/to/vsearch \
python -m uvicorn app.main:app --port 8000
```

`web/data` is not committed. `rsync --delete` would carry the symlink to the
server and replace the real data directory with it.

## Rebuilding the data

`data/web/` is build output, produced by the steps in Step 1. What each script
writes, and why the order matters, is in
[`script/web_export/README.md`](script/web_export/README.md).

## Tests

```bash
node tests/js/preprocess.test.mjs          # the browser path against Python
node tests/js/table.test.mjs               # the count-table reader
cd server && python -m pytest tests -v     # the endpoint's limits and mapping
```

The first is the one that matters. The site re-implements the paper's
preprocessing in JavaScript, and a mistake there produces a plausible wrong
number rather than an error. The second covers the shapes count tables
actually arrive in — the banner `biom convert --to-tsv` writes, a table
transposed in a spreadsheet, a row longer than its header — because the
parser is the first thing a visitor's file meets and a TypeError from inside
it is not something they can act on. See
[`tests/js/README.md`](tests/js/README.md) for what is asserted and the one
place the two languages legitimately differ.

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

## Further reading

- [`deploy/WEB_CONFIG.md`](deploy/WEB_CONFIG.md) — the same deployment in more
  depth, plus operations, known limits and failure modes.
- [`script/web_export/README.md`](script/web_export/README.md) — the build
  pipeline and the two things in it that are easy to get silently wrong.
