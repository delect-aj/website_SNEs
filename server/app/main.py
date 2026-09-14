"""The only server-side endpoint on this site.

Everything else is a static file. This exists for one job: take a rep-seqs
FASTA and say which reference OTU each sequence is, so that a visitor whose
pipeline produces ASVs rather than OTU ids can still score a sample. The
classification itself happens in their browser.

The cost of that job is controlled by how big the reference database is.
Aligning against the 8,850 OTU reference sequences takes well under a second
per sample on one core; aligning against all of SILVA would take minutes. The
database is the vocabulary the model was trained on, so it is also the only
set of sequences that could produce a usable answer.

Nothing is stored. The upload goes to a temporary file, vsearch reads it, and
the file is removed before the response is returned.
"""

import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

MAX_BYTES = 10 * 1024 * 1024
MAX_SEQUENCES = 5000
VSEARCH_TIMEOUT_SECONDS = 120
IDENTITY = "0.97"

# A floor on sequence length, not a quality filter.
#
# `vsearch --usearch_global` scores identity over the aligned region, so a
# 50-base fragment that happens to be a perfect prefix of a reference sequence
# is reported as a 100% match. That is correct as an alignment and misleading
# as an answer: no amplicon workflow produces 50-base rep-seqs, and a fragment
# that short carries too little information for 97% identity to mean anything.
# Real V3-V4 and V4 amplicons are 200-450 bases, so this only rejects input
# that was never going to be comparable to the reference cohort.
MIN_SEQUENCE_LENGTH = 100

# Per-client request budget. A single request already costs a core for about a
# second, so this is deliberately tight: a small server should slow abusive
# clients rather than fall over for everyone else.
RATE_WINDOW_SECONDS = 60
RATE_MAX_REQUESTS = 10

DATABASE = os.environ.get(
    "OTU_REFSEQS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                 "otu_refseqs.fasta"))
# The atlas searches every OTU on the map, not just the model's vocabulary:
# 14,093 sequences against the 8,850 above. Mapping an ASV for the dysbiosis
# score against this one would hand the model ids it has no embedding for, so
# the two stay separate and the caller picks. It sits beside the model's
# database unless told otherwise, so a deployment that sets only OTU_REFSEQS
# finds it without a second variable.
ATLAS_DATABASE = os.environ.get(
    "ATLAS_REFSEQS",
    os.path.join(os.path.dirname(DATABASE), "atlas_refseqs.fasta"))
DATABASES = {"model": DATABASE, "atlas": ATLAS_DATABASE}

# How many equally close references one query may report. The model takes one
# id per ASV, so its database keeps vsearch's first accepted hit. The atlas is
# answering "which OTU is this", and a 250-base V4 read is identical to more
# than one full-length 97% OTU a third of the time; returning only the first
# would name one of them as if it were the answer. Every hit at the top
# identity is reported instead, up to this cap. Measured on 479 V4 fragments of
# atlas sequences: the true source is among the hits for 473 (65% with one
# hit), at about 12 ms a query; exhaustive search finds all 479 at 1.2 s each.
ATLAS_MAX_HITS = 64
VSEARCH = os.environ.get("VSEARCH_BINARY", "vsearch")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Print where vsearch and the database came from at startup.

    A missing binary or a wrong bind-mount path is the usual deployment
    mistake, and it otherwise surfaces as a 503 the first time a visitor
    uploads something.
    """
    print(f"[map] vsearch: {shutil.which(VSEARCH) or 'NOT FOUND'}")
    for name, path in DATABASES.items():
        print(f"[map] {name} database: {path} "
              f"({'present' if os.path.exists(path) else 'MISSING'})")
    yield


app = FastAPI(title="Microbial social niches — reference mapping",
              version="1.0",
              description=__doc__,
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_requests = defaultdict(deque)
_rate_lock = threading.Lock()

# One vsearch at a time. The endpoint runs in a threadpool so that a long
# alignment cannot block the event loop, and this keeps that freedom from
# turning into several vsearch processes competing for the one core.
_vsearch_lock = threading.Semaphore(1)


def rate_limited(client):
    """True when `client` has spent its request budget for this window."""
    now = time.monotonic()
    with _rate_lock:
        # One deque per client that has ever called would grow without bound,
        # so the map is swept once it is larger than a site this size should
        # ever see. Entries whose window has passed are free to drop.
        if len(_requests) > 10_000:
            for other in [key for key, window in _requests.items()
                          if not window
                          or now - window[-1] > RATE_WINDOW_SECONDS]:
                del _requests[other]

        history = _requests[client]
        while history and now - history[0] > RATE_WINDOW_SECONDS:
            history.popleft()
        if len(history) >= RATE_MAX_REQUESTS:
            return True
        history.append(now)
        return False


def read_capped(upload):
    """Read an upload, refusing anything past the size limit.

    Reading in chunks rather than calling `.read()` means an oversized body is
    rejected after the first megabyte instead of after it has all been
    buffered.

    Raises
    ------
    HTTPException
        413 when the body exceeds `MAX_BYTES`.
    """
    chunks = []
    total = 0
    while True:
        chunk = upload.file.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"That file is larger than {MAX_BYTES // (1024 * 1024)} MB. "
                       f"Send one sample's rep-seqs, not a whole run.")
        chunks.append(chunk)
    return b"".join(chunks)


def parse_fasta(text):
    """Split a FASTA body into ``[(id, sequence)]``.

    Raises
    ------
    HTTPException
        400 when the body is empty, is not FASTA, holds a record with no
        sequence, holds a sequence below `MIN_SEQUENCE_LENGTH`, or holds more
        records than the limit.
    """
    records = []
    header = None
    lines = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(lines)))
            header = line[1:].split()[0] if len(line) > 1 else ""
            lines = []
        else:
            if header is None:
                raise HTTPException(
                    status_code=400,
                    detail="This is not FASTA: the file has sequence data "
                           "before any '>' header line.")
            lines.append(line)

    if header is not None:
        records.append((header, "".join(lines)))

    if not records:
        raise HTTPException(status_code=400,
                            detail="No sequences found. A rep-seqs FASTA is "
                                   "expected: one '>' header per sequence.")

    empty = [name for name, sequence in records if not sequence]
    if empty:
        raise HTTPException(
            status_code=400,
            detail=f"{len(empty)} record(s) have a header but no sequence, "
                   f"starting with {empty[0]!r}.")

    # Count before length: a file that is over the limit should be told so
    # whatever its sequences look like.
    if len(records) > MAX_SEQUENCES:
        raise HTTPException(
            status_code=413,
            detail=f"{len(records)} sequences, over the limit of "
                   f"{MAX_SEQUENCES}. One sample's rep-seqs is expected.")

    short = [(name, len(sequence)) for name, sequence in records
             if len(sequence) < MIN_SEQUENCE_LENGTH]
    if short:
        raise HTTPException(
            status_code=400,
            detail=f"{len(short)} sequence(s) are shorter than "
                   f"{MIN_SEQUENCE_LENGTH} bases, the shortest being "
                   f"{short[0][0]!r} at {short[0][1]}. Short fragments align "
                   f"to a reference by chance as well as by descent, so they "
                   f"are refused rather than matched. Rep-seqs from DADA2 or "
                   f"QIIME2 are the expected input.")

    return records


def run_vsearch(query_path, database, max_hits=1):
    """Run vsearch and return ``{query_id: ([subject_id, ...], percent_identity)}``.

    With ``max_hits`` above one, every subject tied at the best identity is
    listed, in the order vsearch reports them.

    Raises
    ------
    HTTPException
        503 when the databases are missing or vsearch is not installed, 504 on
        timeout, 500 when vsearch fails for any other reason.
    """
    if not os.path.exists(database):
        raise HTTPException(
            status_code=503,
            detail="The reference database is not installed on this server.")

    command = [VSEARCH, "--usearch_global", query_path,
               "--db", database, "--id", IDENTITY,
               "--maxaccepts", str(max_hits), "--maxhits", str(max_hits),
               "--threads", "1", "--blast6out", "-"]
    if max_hits > 1:
        command[-2:-2] = ["--maxrejects", str(max_hits), "--top_hits_only"]

    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=VSEARCH_TIMEOUT_SECONDS)
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="vsearch is not available on this server.")
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail=f"Mapping took longer than {VSEARCH_TIMEOUT_SECONDS} seconds "
                   f"and was stopped.")

    if completed.returncode != 0:
        raise HTTPException(status_code=500,
                            detail=f"vsearch exited with {completed.returncode}.")

    mapping = {}
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) >= 3:
            subjects, _ = mapping.setdefault(fields[0], ([], float(fields[2])))
            subjects.append(fields[1])
    return mapping


@app.get("/health")
async def health():
    """Report readiness: the database has to be present for /map to work.

    Async so that it never queues in the threadpool behind a running vsearch:
    a readiness check has to answer while the service is busy, which is when
    it is worth asking.
    """
    installed = os.path.exists(DATABASE)
    return {
        "status": "ok" if installed else "degraded",
        "database": DATABASE,
        "database_present": installed,
        "atlas_database": ATLAS_DATABASE,
        "atlas_database_present": os.path.exists(ATLAS_DATABASE),
        "identity": IDENTITY,
        "limits": {"max_bytes": MAX_BYTES, "max_sequences": MAX_SEQUENCES,
                   "requests_per_minute": RATE_MAX_REQUESTS},
    }


@app.post("/map")
def map_sequences(request: Request,
                  rep_seqs: UploadFile = File(...),
                  db: Literal["model", "atlas"] = "model"):
    """Map rep-seqs headers to reference OTU ids at 97% identity.

    ``db=model`` (the default) searches the dysbiosis model's vocabulary;
    ``db=atlas`` searches every OTU in the atlas.

    Sync on purpose: reading the upload and running vsearch are both blocking,
    and a `def` endpoint is run in the threadpool, so a slow alignment does not
    stall every other request on the worker -- `/health` included.
    """
    client = request.client.host if request.client else "unknown"
    if rate_limited(client):
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. The limit is {RATE_MAX_REQUESTS} per "
                   f"minute; please wait and try again.")

    body = read_capped(rep_seqs)
    records = parse_fasta(body.decode("utf-8", errors="replace"))

    handle, path = tempfile.mkstemp(suffix=".fasta", prefix="map-")
    try:
        with os.fdopen(handle, "w") as temporary:
            for name, sequence in records:
                temporary.write(f">{name}\n{sequence}\n")
        with _vsearch_lock:
            hits = run_vsearch(path, DATABASES[db],
                               ATLAS_MAX_HITS if db == "atlas" else 1)
    finally:
        # Removed whether or not vsearch succeeded; nothing is kept.
        os.unlink(path)

    return JSONResponse({
        "mapping": {query: otus[0] for query, (otus, _) in hits.items()},
        "hits": {query: otus for query, (otus, _) in hits.items()},
        "identity": {query: identity for query, (_, identity) in hits.items()},
        "mapped": len(hits),
        "total": len(records),
    })
