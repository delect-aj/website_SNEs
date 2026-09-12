# Deployment plan

Everything needed to put this site on a small public server, and the checks
that tell you it went correctly.

---

## 1. What is being deployed

Two pieces, and only one of them is a running service.

```
                    visitor's browser
                    ├─ all model inference (onnxruntime-web, WebAssembly)
                    ├─ all preprocessing
                    └─ all rendering
                             │
   nginx (host) ─────────────┼──────────────────────────────────────
   ├─ /, /atlas, /dysbiosis, /download, /cite      static files
   ├─ /data/*                                      static files
   └─ /map ──────────────► map service (localhost:8000)
                           └─ vsearch against data/otu_refseqs.fasta
```

The split is what makes a 1-core host enough. The browser pays for the neural
network — 4.7 MB downloaded once, then cached — and the server only does the
one thing a browser cannot: aligning a visitor's sequences against the
reference OTUs with vsearch. A few hundred sequences take well under a second.

The reference database holds **8,850 sequences**, not the 14,093 the embedding
file has. The model's vocabulary is a different 14,019 ids that overlap the
embedding in 8,850 places, and only those 8,850 have both a sequence and a
trained vector. Mapping to anything else would produce an OTU the model scores
as absent, which is a silently wrong answer rather than an error.

---

## 2. Sizing

| Resource | Minimum | Notes |
|---|---|---|
| vCPU | 1 | vsearch is single-threaded and the container is capped at one core |
| RAM | 2 GB | nginx ~50 MB, map service ~150 MB under load, page cache the rest |
| Disk | 20 GB | site 45 MB, reference database 13 MB, OS and logs the rest |
| Bandwidth | — | see below |

Bandwidth, per visitor who uses a page:

| Page | First visit | Repeat |
|---|---|---|
| Home, download, cite | < 50 kB | cached |
| Atlas | ~4.6 MB gzipped | cached |
| Dysbiosis | ~18 MB gzipped, of which 10.6 MB is the ONNX Runtime WebAssembly | cached |

The dysbiosis page is heavy and that is a deliberate trade: the alternative is
running inference on the server, which multiplies the required cores by
however many people visit at once. The runtime is shared between all visitors
through the browser cache, so the cost is paid per visitor once, not per
sample. If that is unacceptable, quantising the graph to int8 takes the 4.7 MB
model to roughly 1.2 MB at the cost of a small accuracy change — that is a
change to make deliberately, with the regression tests re-run, not by default.

---

## 3. Building the artefacts

Run once on the machine that has the research data, not on the server. The
export scripts read the BIOM tables, the fold checkpoints and the embedding
text files and write everything the site serves.

```bash
cd website_SNEs

# A virtual environment with torch, onnx, onnxruntime, biom, h5py, scipy.
python -m venv --system-site-packages .venv-export
.venv-export/bin/pip install "onnx==1.15.0" "onnxruntime==1.16.3"

# 1. The atlas arrays. Produces umap.f32.bin, both neighbour lists, otus.json
#    and meta.json. Run the notebook as written; the web build only reads it.
jupyter nbconvert --execute script/atlas_export.ipynb

# 2. The dysbiosis model, vocabulary, reference scores and examples.
#    Verifies itself: it reproduces the training run's pred_test.csv before
#    writing anything, and checks onnxruntime against PyTorch afterwards.
.venv-export/bin/python script/web_export/export_dysbiosis.py

# 3. The trait probabilities behind the confidence bands and the BacDive
#    overrides. Refits the forests with the notebook's own seed.
.venv-export/bin/python script/web_export/export_traits.py

# 4. Quantised similarities, the download formats, the vsearch database and
#    the manifest. Run last: it reads what 1-3 wrote and rewrites meta.json.
.venv-export/bin/python script/web_export/export_assets.py \
    --site-url https://microbiome.example.org
```

`--site-url` is the public origin. It goes into the TensorFlow Embedding
Projector config, so it must match the domain the site is served from.

### What gets copied to the server

```
web/                        ->  /srv/microbial/site
data/web/                   ->  /srv/microbial/site/data
data/server/otu_refseqs.fasta ->  /srv/microbial/data/otu_refseqs.fasta
server/                     ->  /srv/microbial/server   (for the container build)
deploy/                     ->  configuration
```

`web/` is the document root. `data/web/` sits inside it under `/data/`. The
reference FASTA deliberately sits outside it — it is 13 MB that no browser
ever needs.

```bash
rsync -av --delete web/ data/web/  server:  # see the runbook below
```

---

## 4. Server preparation

Ubuntu 22.04 or 24.04, Debian 12, or any distribution with nginx ≥ 1.18.

```bash
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx
```

Then choose one of the two ways to run the map service.

### Option A — Docker Compose

```bash
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker "$USER"     # then log out and back in
cd /srv/microbial
docker compose -f deploy/docker-compose.yml up -d --build
curl -s localhost:8000/health | python3 -m json.tool
```

### Option B — systemd, no Docker

Needs vsearch, which most distributions do not package. Conda is the reliable
source:

```bash
sudo useradd --system --home /srv/microbial --shell /usr/sbin/nologin microbial
sudo -u microbial python3 -m venv /srv/microbial/venv
sudo -u microbial /srv/microbial/venv/bin/pip install -r /srv/microbial/server/requirements.txt

# vsearch, for the whole system
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
curl -s localhost:8000/health | python3 -m json.tool
```

Check `/health` says `"database_present": true`. If it does not, the bind
mount or `OTU_REFSEQS` path is wrong, and `/map` will return 503.

---

## 5. nginx and TLS

The provided `nginx.conf` needs one line added to the `http` block of
`/etc/nginx/nginx.conf`, because rate-limit zones cannot be declared inside a
server block:

```nginx
limit_req_zone $binary_remote_addr zone=map_limit:10m rate=10r/m;
```

The supplied config has an HTTPS block referencing a certificate that does not
exist on a fresh host, so installing it first makes `nginx -t` fail. Go in two
phases, with the certificate obtained between them:

```bash
sudo mkdir -p /var/www/certbot

# Phase 1: HTTP only. Enough for nginx to start and for certbot to answer the
# ACME challenge. DNS must already point at this host.
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

# Phase 2: the real config, which now has a certificate to point at.
sudo cp deploy/nginx.conf /etc/nginx/sites-available/microbial-embeddings
sudo sed -i 's|SITE_ROOT|/srv/microbial/site|' \
  /etc/nginx/sites-available/microbial-embeddings
sudo sed -i 's|microbiome.example.org|your.domain|g' \
  /etc/nginx/sites-available/microbial-embeddings
sudo nginx -t && sudo systemctl reload nginx
```

Certbot installed a renewal timer when it issued the certificate. Check it:
`sudo systemctl list-timers | grep certbot`. Renewal reuses the webroot path
above, which the deployed config keeps serving.

One header is load-bearing and easy to miss: `.wasm` must be served as
`application/wasm`, because onnxruntime-web instantiates it with the streaming
API, which refuses anything else. The symptom is a dysbiosis page that does
nothing and logs nothing useful. The config handles it in its own location
block rather than in a `types` block, which would replace the inherited MIME
map instead of extending it.

### Firewall

```bash
sudo ufw allow 22,80,443/tcp
sudo ufw enable
```

Port 8000 is bound to 127.0.0.1 by both deployment options and must stay that
way — vsearch behind no rate limit and no TLS is not something to expose.

---

## 6. Verifying the deployment

Run these in order. Each one catches a different class of mistake.

```bash
# 1. The service is up and can see its database.
curl -s https://your.domain/health | python3 -m json.tool
#    expect: "database_present": true

# 2. The static site is complete: every file in the manifest resolves.
curl -s https://your.domain/data/manifest.json \
  | python3 -c "import json,sys; [print(f['file']) for f in json.load(sys.stdin)['files']]" \
  | while read -r f; do
      code=$(curl -s -o /dev/null -w '%{http_code}' "https://your.domain/data/$f")
      [ "$code" = 200 ] || echo "MISSING $f ($code)"
    done

# 3. Compression is on where it matters. otus.json should be ~0.9 MB.
curl -s -H 'Accept-Encoding: gzip' -o /dev/null -w '%{size_download}\n' \
  https://your.domain/data/otus.json

# 4. /map round-trips: sequences taken from the database must come back as
#    themselves, not merely as something.
head -c 40000 data/server/otu_refseqs.fasta > /tmp/probe.fasta
curl -s -F "rep_seqs=@/tmp/probe.fasta" https://your.domain/map \
  | python3 -c "
import json, sys
payload = json.load(sys.stdin)
print(payload['mapped'], 'of', payload['total'])
bad = [(q, s) for q, s in payload['mapping'].items() if q != s]
print('identity mismatches:', len(bad), bad[:3])
assert payload['mapped'] == payload['total'] and not bad"

# 5. Errors are 4xx, not 5xx.
printf 'not a fasta\n' > /tmp/bad.txt
curl -s -o /dev/null -w '%{http_code}\n' \
  -F "rep_seqs=@/tmp/bad.txt" https://your.domain/map     # expect 400
```

Then open the site and walk the golden path by hand:

1. `/atlas` loads, the scatter draws 14,093 points, the colour-by selector
   changes them, and searching a genus fills the card.
2. On a card, "Show the labelled distribution" draws two groups of dots and a
   marker, and a trait with AUC below 0.65 sits folded at the bottom.
3. `/dysbiosis` → *Run an example sample* → a percentile appears with the
   cohort distribution under it, and a contributing taxon links into `/atlas`.
4. The same page with a real rep-seqs FASTA plus counts goes through `/map`
   and produces a score. The mapping line should report a plausible fraction
   of sequences mapped; if it is near zero, the sequences are not SILVA 138.2
   97% OTU reps and the sample is not comparable to the cohort.

---

## 7. The regression tests

These run on the build machine, not the server, and they are the reason the
numbers on the page can be trusted.

```bash
# The Python pipeline reproduces the training run, and the exported ONNX
# reproduces the Python. Both are asserted inside the export itself.
.venv-export/bin/python script/web_export/export_dysbiosis.py

# The browser's preprocessing against Python's, over the whole reference
# cohort. This is the test that catches a mis-ported rank normalization.
node tests/js/preprocess.test.mjs

# The endpoint's limits and its round trip.
cd server && python -m pytest tests -v
```

The design document's requirement was a difference below 1e-4 between the
Python pipeline and the browser path. See `tests/js/README.md` for what was
measured and why the tolerance is set where it is.

---

## 8. Operations

### Updating the site

```bash
cd website_SNEs
git pull
rsync -av --delete web/ server:/srv/microbial/site/
rsync -av --delete data/web/ server:/srv/microbial/site/data/
sudo systemctl reload nginx     # only needed if the config changed
```

The rsync `--delete` is safe here because both directories are build output.
Never point it at `data/server/`.

### Rebuilding the reference database

Only needed if the vocabulary changes:

```bash
.venv-export/bin/python script/web_export/export_assets.py
rsync -av data/server/otu_refseqs.fasta server:/srv/microbial/data/
curl -s localhost:8000/health    # on the server
```

### Logs

```bash
docker compose -f deploy/docker-compose.yml logs -f map   # Option A
journalctl -u microbial-map -f                            # Option B
sudo tail -f /var/log/nginx/access.log
```

nginx logs are JSON-free by default. To find how `/map` is being used, add
`log_format` with `$request_time` and `$status`, or count 429s:
`grep -c ' 429 ' /var/log/nginx/access.log`.

### Backups

Nothing on the server is stateful. The only thing worth keeping is the build
output on the lab machine and the git repository. A rebuild from scratch is
one `rsync` and, at most, one export run.

---

## 9. Known limits, and what to tell visitors

**8,850 usable OTUs, not 14,093.** The atlas shows all 14,093 embedded OTUs
because the embedding covers them; the classifier knows 8,850. A sample whose
community falls mostly outside that set will score with most positions masked,
and the page reports how many.

**The model was trained across the reference cohort.** Its leave-one-disease-out
AUC is 0.64, against 0.80 for the cohort it trained on. A sample from a
condition outside the thirteen will sit closer to the middle than it should.
The site states this on the result and on `/cite`; do not remove those lines to
make the number look better.

**The percentile is not a probability.** Nothing on the site should ever say
"risk", "probability of disease" or "diagnosis" about a sample. The design
document rules these out, and the pages enforce it in their copy.

**HDF5 BIOM files are not read in the browser.** The page says so and
suggests `biom convert --to-tsv`, or the FASTA route, which the server maps.
Adding a WebAssembly HDF5 reader would fix it at the cost of another megabyte
of runtime, which did not look like the right trade for the first version.

**The Embedding Projector link needs CORS and a public address.** Its config
points at this site's `/data/download/*.tsv`. Those files are served with
`Access-Control-Allow-Origin: *`; if the header is dropped, the Projector shows
an empty canvas.
