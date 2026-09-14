"""Endpoint tests for /map.

The round-trip test is the one that matters: take sequences straight out of
the reference database, submit them, and assert every one comes back as the
OTU it came from. A limit test that passes while the mapping is wrong would be
worse than no test at all.

Run with the reference database and vsearch present:

    OTU_REFSEQS=../data/server/otu_refseqs.fasta \\
    VSEARCH_BINARY=/path/to/vsearch \\
    python -m pytest server/tests -v
"""

import os
import random
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from app import main  # noqa: E402

client = TestClient(main.app)

DATABASE = main.DATABASE
have_database = os.path.exists(DATABASE)
have_vsearch = main.shutil.which(main.VSEARCH) is not None

needs_mapping = pytest.mark.skipif(
    not (have_database and have_vsearch),
    reason=f"needs the reference database ({DATABASE}) and vsearch")

needs_atlas = pytest.mark.skipif(
    not (have_database and os.path.exists(main.ATLAS_DATABASE) and have_vsearch),
    reason=f"needs both databases ({DATABASE}, {main.ATLAS_DATABASE}) and vsearch")


@pytest.fixture(autouse=True)
def fresh_rate_limit():
    """Every test starts with a full request budget.

    The limiter is keyed by client address and TestClient always has the same
    one, so without this the eleventh request of the session is a 429.
    """
    main._requests.clear()
    yield
    main._requests.clear()


def read_sequences(limit=40, database=DATABASE, skip=()):
    """Pull a few records straight out of a reference database."""
    records = []
    header = None
    sequence = []
    with open(database) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header and header not in skip:
                    records.append((header, "".join(sequence)))
                    if limit and len(records) >= limit:
                        return records
                header = line[1:].split()[0]
                sequence = []
            else:
                sequence.append(line)
    if header and header not in skip:
        records.append((header, "".join(sequence)))
    return records


def as_fasta(records):
    return "\n".join(f">{name}\n{sequence}" for name, sequence in records)


def post(content, filename="rep_seqs.fasta", db=None):
    return client.post("/map", params={"db": db} if db else None,
                       files={"rep_seqs": (filename, content, "text/plain")})


def test_health_reports_readiness():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["database_present"] == have_database
    assert payload["atlas_database_present"] == os.path.exists(main.ATLAS_DATABASE)
    assert payload["identity"] == "0.97"
    assert payload["limits"]["max_sequences"] == 5000


def test_empty_file_is_rejected():
    response = post("")
    assert response.status_code == 400
    assert "No sequences" in response.json()["detail"]


def test_plain_text_is_rejected():
    response = post("this is not a fasta file\njust some words\n")
    assert response.status_code == 400
    assert "not FASTA" in response.json()["detail"]


def test_header_without_sequence_is_rejected():
    response = post(">asv_1\n>asv_2\nACGT\n")
    assert response.status_code == 400
    assert "no sequence" in response.json()["detail"]


def test_oversized_body_is_rejected():
    payload = ">asv_1\n" + "A" * (main.MAX_BYTES + 1024) + "\n"
    response = post(payload)
    assert response.status_code == 413
    assert "larger than" in response.json()["detail"]


def test_too_many_sequences_are_rejected():
    sequence = "ACGT" * 50        # above the length floor, so only the count fires
    payload = "\n".join(f">asv_{i}\n{sequence}"
                        for i in range(main.MAX_SEQUENCES + 1))
    response = post(payload)
    assert response.status_code == 413
    assert "over the limit" in response.json()["detail"]


@needs_mapping
def test_known_sequences_map_back_to_their_own_otu():
    records = read_sequences(limit=40)
    assert records, "the reference database is empty"

    response = post(as_fasta(records))
    assert response.status_code == 200
    payload = response.json()

    assert payload["total"] == len(records)
    assert payload["mapped"] == len(records), (
        "sequences taken verbatim from the database must map back to "
        "themselves; anything less means the database or the identity "
        "threshold changed")

    for name, _ in records:
        assert payload["mapping"][name] == name
        assert payload["identity"][name] == 100.0


def test_an_unknown_database_is_rejected():
    response = post(">asv_1\n" + "ACGT" * 50 + "\n", db="silva")
    assert response.status_code == 422


def atlas_only_records(limit):
    """Atlas sequences for OTUs the model has no row for."""
    model = {name for name, _ in read_sequences(limit=None)}
    return read_sequences(limit=limit, database=main.ATLAS_DATABASE, skip=model)


@needs_atlas
def test_the_atlas_lists_every_equally_close_otu(tmp_path, monkeypatch):
    """A read shared by several OTUs is reported as all of them.

    Duplicating a reference under a second name makes a tie that cannot be
    down to chance, so both names have to come back, at the same identity.
    """
    records = read_sequences(limit=20, database=main.ATLAS_DATABASE)
    name, sequence = records[0]
    database = tmp_path / "tied.fasta"
    database.write_text(as_fasta(records + [("twin", sequence)]) + "\n")
    monkeypatch.setitem(main.DATABASES, "atlas", str(database))

    payload = post(as_fasta([("read", sequence[:250])]), db="atlas").json()
    assert sorted(payload["hits"]["read"]) == sorted([name, "twin"])
    assert payload["mapping"]["read"] in (name, "twin")
    assert payload["identity"]["read"] == 100.0


@needs_atlas
def test_the_atlas_database_reaches_otus_outside_the_vocabulary():
    """The reason the atlas has its own database.

    These OTUs are on the map and have trait cards, but no row in the model,
    so the default database cannot return them.
    """
    records = atlas_only_records(limit=30)
    assert records, "the atlas database holds nothing beyond the vocabulary"

    payload = post(as_fasta(records), db="atlas").json()
    assert payload["mapped"] == payload["total"] == len(records)
    for name, _ in records:
        assert payload["mapping"][name] == name
        assert payload["identity"][name] == 100.0

    default = post(as_fasta(records)).json()
    assert all(len(hits) == 1 for hits in default["hits"].values())
    assert not any(default["mapping"].get(name) == name for name, _ in records)


def test_sequences_below_the_length_floor_are_rejected():
    """A 50-base fragment must not come back as a confident match.

    vsearch scores identity over the aligned region, so a fragment that is a
    perfect prefix of a reference is reported as 100% identical. Correct as an
    alignment, misleading as an answer; the length floor is what stops it.
    """
    records = read_sequences(limit=5)
    fragments = [(name, sequence[:50]) for name, sequence in records]
    response = post(as_fasta(fragments))
    assert response.status_code == 400
    assert "shorter than" in response.json()["detail"]


@needs_mapping
def test_fragments_above_the_floor_map_to_their_own_otu():
    """A 200-base fragment of a reference is a legitimate match.

    The reference sequences are full-length SSU, while real ASVs are 200-450
    bases of one variable region, so this is the ordinary case rather than an
    edge case: a fragment aligns to a substring of its own reference and
    should be reported as that reference.
    """
    records = read_sequences(limit=20)
    fragments = [(name, sequence[:200]) for name, sequence in records]
    response = post(as_fasta(fragments))
    assert response.status_code == 200
    payload = response.json()
    assert payload["mapped"] == payload["total"]
    for name, _ in records:
        assert payload["mapping"][name] == name


@needs_mapping
def test_unknown_sequences_map_to_nothing():
    rng = random.Random(0)
    payload = "\n".join(
        f">random_{i}\n" + "".join(rng.choice("ACGT") for _ in range(200))
        for i in range(20))
    response = post(payload)
    assert response.status_code == 200
    body = response.json()
    # Random sequence occasionally matches something short by chance; what
    # must not happen is most of it mapping.
    assert body["mapped"] <= 2
    assert body["total"] == 20


def test_rate_limit_returns_429_not_500():
    client_ = TestClient(main.app)
    main._requests.clear()
    statuses = []
    for _ in range(main.RATE_MAX_REQUESTS + 3):
        statuses.append(client_.post(
            "/map", files={"rep_seqs": ("x.fasta", "", "text/plain")}).status_code)
    main._requests.clear()
    assert 429 in statuses, "the limiter never fired"
    assert 500 not in statuses
