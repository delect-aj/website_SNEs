"""Write the example input files the site offers for download.

A visitor who has never exported an OTU table cannot guess what the upload
boxes want, so each input route gets a small real file to open or run:

    examples/atlas_asv_example.fasta   two 300-base V4 reads for the atlas search
    examples/otu_table_example.tsv     two samples keyed by SILVA 138.2 OTU id
    examples/rep_seqs_example.fasta    the same two samples as amplicon sequences
    examples/asv_table_example.tsv     ... and their ASV count table

The two samples are the CRC control and case the page already offers as
one-click examples, so a file uploaded here scores the same community.

Reads are cut the way an amplicon run would produce them: the 300 bases that
follow the 515F primer site of the OTU's reference sequence. The atlas reads
also carry two substitutions each, so the search reports a realistic identity
below 100% rather than an exact copy of the database. Both were checked with
vsearch against `atlas_refseqs.fasta` using the server's flags: each returns
exactly one OTU, its source.

For the dysbiosis route, OTUs whose reference sequence lacks the primer site
are left out, and OTUs whose reads are identical are merged into one ASV, as a
denoiser would. The FASTA route therefore sees a slightly smaller community than
the table route and its score can differ a little; that is how real data behave.

Standard library only. Run after export_assets.py, because it reads the vsearch
databases that script writes and refreshes the manifest it writes:

    python3 script/web_export/export_examples.py
"""

import hashlib
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB = os.path.join(REPO, "data", "web")
SERVER = os.path.join(REPO, "data", "server")

PRIMER_515F = re.compile("GTG[CT]CAGC[AC]GCCGCGGTAA")
READ_LENGTH = 300

# (OTU id, genus) for the atlas reads. Both are genome-linked, so the card
# that opens shows labelled traits rather than only predictions.
ATLAS_READS = [("DQ805015.1.1370", "Akkermansia"), ("DQ803549.1.1391", "Blautia")]
# Positions within the read and the base swapped in: transitions, two per read.
SUBSTITUTIONS = (100, 200)
TRANSITION = {"A": "G", "G": "A", "C": "T", "T": "C"}

# Column names are "<sample id>_CRC_<group>", the id read from each file.
SAMPLES = [("crc_control.json", "CRC_control"), ("crc_case.json", "CRC_case")]


def read_fasta(path):
    sequences, name = {}, None
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].split()[0]
                sequences[name] = []
            elif name:
                sequences[name].append(line.upper())
    return {name: "".join(parts) for name, parts in sequences.items()}


def amplicon(sequence):
    """The READ_LENGTH bases after the 515F site, or None when there is no site."""
    match = PRIMER_515F.search(sequence)
    if not match or len(sequence) < match.end() + READ_LENGTH:
        return None
    return sequence[match.end():match.end() + READ_LENGTH]


def write_table(path, rows, sample_names):
    """biom-convert TSV: the banner line, then `#OTU ID` and one column per sample."""
    with open(path, "w") as handle:
        handle.write("# Constructed from biom file\n")
        handle.write("\t".join(["#OTU ID"] + sample_names) + "\n")
        for feature, counts in rows:
            handle.write("\t".join([feature] + [str(c) for c in counts]) + "\n")


def write_atlas_reads(out_dir):
    atlas = read_fasta(os.path.join(SERVER, "atlas_refseqs.fasta"))
    with open(os.path.join(out_dir, "atlas_asv_example.fasta"), "w") as handle:
        for number, (otu, genus) in enumerate(ATLAS_READS, start=1):
            read = list(amplicon(atlas[otu]))
            for position in SUBSTITUTIONS:
                read[position] = TRANSITION[read[position]]
            handle.write(f">ASV_{number} {genus}, V4 region, 300 bp\n{''.join(read)}\n")


def write_sample_tables(out_dir):
    names = []
    counts = []
    for filename, suffix in SAMPLES:
        with open(os.path.join(out_dir, filename)) as handle:
            record = json.load(handle)
        names.append(f"{record['sample_id']}_{suffix}")
        counts.append(record["counts"])
    otus = sorted(set().union(*counts), key=lambda otu: (-sum(c.get(otu, 0) for c in counts), otu))

    write_table(os.path.join(out_dir, "otu_table_example.tsv"),
                [(otu, [c.get(otu, 0) for c in counts]) for otu in otus], names)

    # One ASV per distinct read, first seen in abundance order.
    reference = read_fasta(os.path.join(SERVER, "otu_refseqs.fasta"))
    asvs = {}
    for otu in otus:
        read = amplicon(reference.get(otu, ""))
        if read is None:
            continue
        totals = asvs.setdefault(read, [0] * len(counts))
        for column, sample in enumerate(counts):
            totals[column] += sample.get(otu, 0)

    rows = []
    with open(os.path.join(out_dir, "rep_seqs_example.fasta"), "w") as handle:
        for number, (read, totals) in enumerate(asvs.items(), start=1):
            handle.write(f">ASV_{number:03d}\n{read}\n")
            rows.append((f"ASV_{number:03d}", totals))
    write_table(os.path.join(out_dir, "asv_table_example.tsv"), rows, names)
    return len(otus), len(asvs)


def refresh_manifest(web_dir):
    """Recompute sizes and hashes the way export_assets.py does, keeping its other keys."""
    path = os.path.join(web_dir, "manifest.json")
    with open(path) as handle:
        manifest = json.load(handle)
    files = []
    for root, _, names in os.walk(web_dir):
        for name in sorted(names):
            full = os.path.join(root, name)
            relative = os.path.relpath(full, web_dir)
            if relative == "manifest.json":
                continue
            with open(full, "rb") as handle:
                sha = hashlib.sha256(handle.read()).hexdigest()
            files.append({"file": relative, "bytes": os.path.getsize(full), "sha256": sha})
    manifest["files"] = sorted(files, key=lambda entry: entry["file"])
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=1)


def main():
    out_dir = os.path.join(WEB, "examples")
    write_atlas_reads(out_dir)
    n_otus, n_asvs = write_sample_tables(out_dir)
    refresh_manifest(WEB)
    print(f"  atlas: {len(ATLAS_READS)} reads; samples: {n_otus} OTUs, "
          f"{n_asvs} ASVs with a 515F site")


if __name__ == "__main__":
    main()
