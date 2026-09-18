"""Post-process the atlas arrays and build the download page's artefacts.

Run this after `script/atlas_export.ipynb` and after `export_dysbiosis.py`.
It reads what those wrote -- it never recomputes a layout -- and adds the
things the site serves but the research notebooks have no reason to produce.

    nbr_sne_sim.f16.bin     cosine similarity, float16
    sne.f16.bin             the embedding matrix, float16
    download/*.tsv          vectors and metadata for the TF Projector
    download/*.txt.gz       GloVe text, for gensim (no_header=True)
    download/manifest.json  sizes and checksums for the download page

The phylogenetic neighbours are not here: `export_phylo_neighbours.py` reads
the SILVA tree and writes them as float16 itself.

Why halve the similarities: the array is 2.8 MB as float32 and compress
badly, because a cosine to four decimals is close to incompressible. One byte
per value was the first attempt and is not enough resolution -- see
`quantise_similarity` for what that costs on these numbers.

The write into `meta.json` is additive: every key the notebook put there is
kept, and re-running this script replaces the two entries it owns rather than
duplicating them.
"""

import argparse
import gzip
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_export.trait_labels import load_taxonomy  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUANTISED = ["nbr_sne_sim"]


def quantise_similarity(path, out_path):
    """Halve the similarity arrays to float16.

    One byte per value was the first attempt, scaled over each array's range,
    and it fails on this data. The phylogenetic similarities are not spread
    out: a single row's fifty nearest relatives can all sit between 0.992 and
    1.000, where a 1/256 step of the array's own range leaves three
    distinguishable values and the column reads "1.000" fifty times. A cosine
    to a fourth decimal is also not compressible, which is why these files
    barely shrink under gzip.

    Float16 keeps about three significant digits, so the resolution near 1.0
    is 5e-4 and the column stays readable, for 1.4 MB per array against 2.8 MB
    for float32. The browser already has a half-to-float decoder for the
    embedding table.

    Returns
    -------
    float
        The largest absolute error introduced.
    """
    values = np.fromfile(path, dtype=np.float32)
    if values.min() < -1.001 or values.max() > 1.001:
        raise ValueError(f"{path}: values outside the cosine range "
                         f"[{values.min()}, {values.max()}]")

    halves = values.astype(np.float16)
    halves.tofile(out_path)
    return float(np.abs(halves.astype(np.float32) - values).max())


def write_embedding(script_dir, out_dir):
    """Write the SNE matrix as float16, plus the text formats for download."""
    ids = []
    vectors = []
    with open(os.path.join(script_dir, "social_niche_embedding_100.txt")) as handle:
        for line in handle:
            fields = line.rstrip("\n").split(" ")
            if len(fields) <= 2:
                continue                      # the dimension header, if any
            ids.append(fields[0])
            vectors.append([float(v) for v in fields[1:]])

    matrix = np.array(vectors, dtype=np.float32)
    keep = [i for i, otu in enumerate(ids) if otu != "<unk>"]
    ids = [ids[i] for i in keep]
    matrix = matrix[keep]
    matrix.astype(np.float16).tofile(os.path.join(out_dir, "sne.f16.bin"))

    download = os.path.join(out_dir, "download")
    os.makedirs(download, exist_ok=True)

    # TensorFlow Embedding Projector: values and metadata, tab separated, and
    # the config that points at them.
    with open(os.path.join(download, "projector_vectors.tsv"), "w") as handle:
        for row in matrix:
            handle.write("\t".join(f"{v:.6g}" for v in row))
            handle.write("\n")

    return ids, matrix, download


def write_projector_metadata(ids, script_dir, path):
    """One row per OTU: id and the seven SILVA ranks, tab separated."""
    taxonomy = load_taxonomy(os.path.join(
        script_dir, "taxmap_slv_ssu_ref_nr_138.2.txt")).reindex(ids)
    columns = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    with open(path, "w") as handle:
        handle.write("\t".join(["otu_id"] + columns) + "\n")
        for otu in ids:
            row = [otu]
            for column in columns:
                value = taxonomy.loc[otu, column[0]]
                row.append("" if value is None or value != value else str(value))
            handle.write("\t".join(row) + "\n")
    return taxonomy


def write_glove(ids, matrix, path):
    """GloVe text format, as the embedding was trained: one OTU per line, no
    header. gensim reads it with `load_word2vec_format(..., no_header=True)`."""
    with gzip.open(path, "wt") as handle:
        for otu, row in zip(ids, matrix):
            handle.write(otu + " " + " ".join(f"{v:.6g}" for v in row) + "\n")


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_reference_sequences(web_dir, fasta_path, out_dir,
                              name="otu_refseqs.fasta", wanted=None):
    """Write a vsearch database, restricted by default to the model's vocabulary.

    The source FASTA holds 14,093 sequences, but the model's vocabulary is a
    different 14,019 ids and the two overlap in only 8,850. Keeping the rest
    would let vsearch return OTU ids that have no row in the model -- and no
    embedding either, since the five thousand ids that are in the vocabulary
    but not in this FASTA are exactly the rows that are all zeros. Mapping to
    them would silently score a sample as if those OTUs were absent, which is
    true but only by accident.

    The atlas passes ``wanted`` as every OTU on the map instead, because an
    ASV search there is after a card, not a model input.

    Returns
    -------
    tuple
        ``(written, missing)`` -- sequences written, and wanted ids with no
        sequence available.
    """
    if wanted is None:
        with open(os.path.join(web_dir, "vocab.json")) as handle:
            wanted = set(json.load(handle)["ids"])

    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, name)

    written = 0
    seen = set()
    with open(fasta_path) as source, open(target, "w") as destination:
        header = None
        sequence = []
        for raw in source:
            line = raw.rstrip("\n")
            if line.startswith(">"):
                if header and header in wanted:
                    destination.write(f">{header}\n{''.join(sequence)}\n")
                    written += 1
                    seen.add(header)
                header = line[1:].split()[0]
                sequence = []
            else:
                sequence.append(line)

        if header and header in wanted:
            destination.write(f">{header}\n{''.join(sequence)}\n")
            written += 1
            seen.add(header)

    return written, len(wanted - seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-dir", default=os.path.join(REPO, "script"))
    parser.add_argument("--out", default=os.path.join(REPO, "data", "web"))
    parser.add_argument("--site-url", default=None,
                        help="public origin, used to build the Projector link")
    args = parser.parse_args()

    # Required rather than defaulted: the origin is written into the manifest
    # and into `download/projector_config.json`, and Google's server fetches
    # the two TSVs from it. A placeholder left in those files is a dead link
    # that nothing else on the site would notice.
    if not args.site_url or not args.site_url.startswith(("http://", "https://")):
        parser.error("--site-url is required and must be the public origin, "
                     "e.g. --site-url https://microbiome.example.org")

    meta_path = os.path.join(args.out, "meta.json")
    with open(meta_path) as handle:
        meta = json.load(handle)

    # The notebook's float32 arrays are the input to this step, and they must
    # not stay in the web root: 2.8 MB each, served, read by nothing. They are
    # moved to a sibling directory rather than deleted so that re-running this
    # script does not require re-running the notebook first.
    keep = os.path.join(os.path.dirname(args.out.rstrip(os.sep)), "atlas_source")
    os.makedirs(keep, exist_ok=True)

    print("halving the neighbour similarities")
    for name in QUANTISED:
        target = os.path.join(args.out, f"{name}.f16.bin")
        key = f"{name}.f16.bin"
        source = os.path.join(args.out, f"{name}.f32.bin")
        if not os.path.exists(source):
            source = os.path.join(keep, f"{name}.f32.bin")

        if os.path.exists(source):
            error = quantise_similarity(source, target)
            os.replace(source, os.path.join(keep, f"{name}.f32.bin"))
            shape = (meta["arrays"].get(f"{name}.f32.bin", {}).get("shape")
                     or [os.path.getsize(target) // 2 // meta["k_neighbours"],
                         meta["k_neighbours"]])
            print(f"  {name}: max error {error:.5f}, source moved to "
                  f"{os.path.relpath(keep, REPO)}")
        elif os.path.exists(target):
            shape = [os.path.getsize(target) // 2 // meta["k_neighbours"],
                     meta["k_neighbours"]]
            print(f"  {name}: already halved, source not present")
        else:
            raise FileNotFoundError(
                f"neither {target} nor a float32 source for it exists; run "
                f"script/atlas_export.ipynb first")

        meta["arrays"][key] = {"dtype": "float16", "shape": shape}

    # Derived from the list, not from the manifest keys: on a second run the
    # keys are already gone but the files are still on disk.
    superseded = [f"{name}.f32.bin" for name in QUANTISED]
    meta["arrays"] = {k: v for k, v in meta["arrays"].items()
                      if k not in superseded and not k.endswith("_sim.u8.bin")}
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=1)

    # The float32 originals stay on disk unless they are removed, and the
    # manifest walks the directory, so the site would serve 5.6 MB that
    # nothing reads. Removing them is what makes data/web equal to what is
    # published; rerunning the notebook regenerates them if they are wanted.
    for name in superseded:
        path = os.path.join(args.out, name)
        if os.path.exists(path):
            os.unlink(path)
            print(f"  removed {name} (superseded by its float16 form)")

    print("writing the embedding matrix")
    ids, matrix, download = write_embedding(args.script_dir, args.out)
    taxonomy = write_projector_metadata(
        ids, args.script_dir, os.path.join(download, "projector_metadata.tsv"))
    write_glove(ids, matrix, os.path.join(download, "sne_vectors.txt.gz"))
    print(f"  {len(ids)} vectors of {matrix.shape[1]} dimensions")

    origin = args.site_url.rstrip("/")
    config = {
        "embeddings": [{
            "tensorName": "SNE — human gut OTUs (14,093 x 100)",
            "tensorShape": [len(ids), matrix.shape[1]],
            "tensorPath": f"{origin}/data/download/projector_vectors.tsv",
            "metadataPath": f"{origin}/data/download/projector_metadata.tsv",
        }]
    }
    with open(os.path.join(download, "projector_config.json"), "w") as handle:
        json.dump(config, handle, indent=1)

    print("writing the vsearch database")
    server_dir = os.path.join(os.path.dirname(args.out.rstrip("/")), "server")
    written, missing = write_reference_sequences(
        args.out, os.path.join(REPO, "data", "otu_seq", "feces_seq_16S_silva.fasta"),
        server_dir)
    print(f"  {written} sequences to {server_dir}/otu_refseqs.fasta; "
          f"{missing} vocabulary OTUs have no sequence in the source FASTA "
          f"and are unreachable through /map")

    with open(os.path.join(args.out, "otus.json")) as handle:
        atlas_ids = {record["id"] for record in json.load(handle)}
    written, missing = write_reference_sequences(
        args.out, os.path.join(REPO, "data", "otu_seq", "feces_seq_16S_silva.fasta"),
        server_dir, name="atlas_refseqs.fasta", wanted=atlas_ids)
    print(f"  {written} sequences to {server_dir}/atlas_refseqs.fasta; "
          f"{missing} atlas OTUs have no sequence and cannot be found by ASV")

    # Everything in the web root except the manifest itself, which cannot
    # carry its own hash.
    manifest = []
    for root, _, names in os.walk(args.out):
        for name in sorted(names):
            path = os.path.join(root, name)
            relative = os.path.relpath(path, args.out)
            if relative == "manifest.json":
                continue
            manifest.append({"file": relative, "bytes": os.path.getsize(path),
                             "sha256": digest(path)})
    manifest.sort(key=lambda entry: entry["file"])
    with open(os.path.join(args.out, "manifest.json"), "w") as handle:
        json.dump({"files": manifest,
                   "projector_url": "https://projector.tensorflow.org/?config="
                                    f"{origin}/data/download/projector_config.json",
                   "note": "Every file here is served from /data/ on this site."},
                  handle, indent=1)

    print(f"{len(manifest)} files listed in manifest.json")
    print(f"Projector: https://projector.tensorflow.org/?config={origin}"
          f"/data/download/projector_config.json")


if __name__ == "__main__":
    main()
