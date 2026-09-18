"""Nearest relatives of every OTU, from the SILVA reference tree.

The card's phylogenetic column used to come from PhyloE, the 100-dimensional
PCA embedding, as a cosine similarity. Those numbers were unreadable: the
nearest relatives of an OTU sit at cosine 1.000 to three decimals, so more than
half the cards printed ten identical values. This reads the tree itself and
stores the **patristic distance** -- the sum of branch lengths along the path
between two leaves, in substitutions per site -- which is a quantity with a
meaning, spread over a usable range, and ordered from the closest relative
outwards.

Writes, in `otus.json` order and K per row:

    nbr_phylo_idx.i16.bin    row numbers of the K nearest relatives
    nbr_phylo_dist.f16.bin   their patristic distances, ascending

and updates `nbr_overlap` in otus.json, the array entries in meta.json and the
checksums in manifest.json.

Exact, not approximate. Every node keeps the K leaves below it that are nearest
to that node; a leaf then walks to the root, and at each ancestor the sibling
subtrees offer exactly those lists. Because every leaf of a sibling subtree is
reached through the same path, ordering by distance from the sibling's root is
ordering by distance from the query, so the sibling's own K nearest are the
only candidates it can contribute. The walk stops as soon as the climb alone
exceeds the current Kth distance.

Standard library only (no dendropy, no ete3):

    python3 script/web_export/export_phylo_neighbours.py
"""

import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_examples import refresh_manifest  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB = os.path.join(REPO, "data", "web")
TREE = os.path.join(REPO, "data", "silva_tree",
                    "SSURefNR99_1200_slv_138_2_subset.tre")

K = 50


def parse_newick(text):
    """Return (parent, length, children, leaf_name) as parallel lists.

    Iterative: the SILVA subset is a deep ladder in places, and a recursive
    descent parser overflows Python's stack on it.
    """
    parent, length, children, leaf_name = [], [], [], []

    def add_node(up):
        parent.append(up)
        length.append(0.0)
        children.append([])
        leaf_name.append(None)
        if up is not None:
            children[up].append(len(parent) - 1)
        return len(parent) - 1

    text = text.strip().rstrip(";")
    root = add_node(None)
    current = root
    token = []
    reading_label = False

    i = 0
    while i < len(text):
        char = text[i]
        if char == "(":
            current = add_node(current)
            token, reading_label = [], False
        elif char in ",)":
            # Close whatever the token held: a leaf name, or a branch length.
            if reading_label:
                length[current] = float("".join(token) or 0)
            elif token:
                leaf_name[current] = "".join(token)
            token, reading_label = [], False
            if char == ",":
                current = add_node(parent[current])
            else:
                current = parent[current]
        elif char == ":":
            if token and not reading_label:
                leaf_name[current] = "".join(token)
            token, reading_label = [], True
        else:
            token.append(char)
        i += 1

    if reading_label:
        length[current] = float("".join(token) or 0)
    return parent, length, children, leaf_name


def postorder(children, root=0):
    """Node indices, children before parents, without recursion."""
    order, stack = [], [root]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(children[node])
    order.reverse()
    return order


def merge(lists, k):
    """The k smallest (distance, leaf) pairs from already-sorted lists."""
    out = []
    for entries in lists:
        out.extend(entries)
    out.sort(key=lambda pair: pair[0])
    return out[:k]


def nearest_relatives(parent, length, children, leaf_name, wanted, k=K):
    """`k` nearest leaves of every wanted leaf, as (distance, leaf) lists."""
    order = postorder(children)

    # 1. Bottom-up: the k leaves nearest to each node, among its descendants.
    below = [None] * len(parent)
    for node in order:
        if not children[node]:
            below[node] = [(0.0, node)] if leaf_name[node] in wanted else []
            continue
        below[node] = merge(
            [[(distance + length[child], leaf) for distance, leaf in below[child]]
             for child in children[node]], k)

    # 2. Top-down for each leaf: climb, taking what each sibling subtree offers.
    result = {}
    for leaf in range(len(parent)):
        name = leaf_name[leaf]
        if name not in wanted:
            continue
        best = []
        node, climbed = leaf, 0.0
        while parent[node] is not None:
            climbed += length[node]
            if len(best) == k and climbed >= best[-1][0]:
                break                      # every remaining leaf is farther
            up = parent[node]
            for sibling in children[up]:
                if sibling == node:
                    continue
                offer = [(climbed + length[sibling] + distance, other)
                         for distance, other in below[sibling]]
                best = merge([best, offer], k)
            node = up
        result[name] = best
    return result


def main():
    with open(os.path.join(WEB, "otus.json")) as handle:
        otus = json.load(handle)
    ids = [record["id"] for record in otus]
    row_of = {otu: i for i, otu in enumerate(ids)}

    with open(TREE) as handle:
        parent, length, children, leaf_name = parse_newick(handle.read())
    leaves = {leaf_name[i] for i in range(len(parent)) if not children[i]}
    missing = set(ids) - leaves
    if missing:
        raise ValueError(f"{len(missing)} atlas OTUs are not leaves of the tree, "
                         f"e.g. {sorted(missing)[:3]}")
    print(f"  tree: {len(parent)} nodes, {len(leaves)} leaves "
          f"({sum(1 for i in range(len(parent)) if children[i] and leaf_name[i])} "
          f"internal nodes carry a support label)")

    neighbours = nearest_relatives(parent, length, children, leaf_name, set(ids))

    idx = bytearray()
    dist = bytearray()
    for otu in ids:
        found = [(d, other) for d, other in neighbours[otu]
                 if leaf_name[other] != otu][:K]
        if len(found) < K:
            raise ValueError(f"{otu}: only {len(found)} relatives found")
        for distance, other in found:
            idx += struct.pack("<h", row_of[leaf_name[other]])
            dist += struct.pack("<e", distance)

    with open(os.path.join(WEB, "nbr_phylo_idx.i16.bin"), "wb") as handle:
        handle.write(idx)
    with open(os.path.join(WEB, "nbr_phylo_dist.f16.bin"), "wb") as handle:
        handle.write(dist)
    stale = os.path.join(WEB, "nbr_phylo_sim.f16.bin")
    if os.path.exists(stale):
        os.remove(stale)

    # nbr_overlap is "how many of the K ecological neighbours are also among the
    # K phylogenetic ones", so it moves with the phylogenetic lists.
    with open(os.path.join(WEB, "nbr_sne_idx.i16.bin"), "rb") as handle:
        sne = struct.unpack(f"<{len(ids) * K}h", handle.read())
    phylo = struct.unpack(f"<{len(ids) * K}h", bytes(idx))
    for i, record in enumerate(otus):
        block = slice(i * K, (i + 1) * K)
        record["nbr_overlap"] = len(set(sne[block]) & set(phylo[block]))
    with open(os.path.join(WEB, "otus.json"), "w") as handle:
        json.dump(otus, handle, separators=(",", ":"))

    meta_path = os.path.join(WEB, "meta.json")
    with open(meta_path) as handle:
        meta = json.load(handle)
    meta["arrays"].pop("nbr_phylo_sim.f16.bin", None)
    meta["arrays"]["nbr_phylo_dist.f16.bin"] = {"dtype": "float16",
                                                "shape": [len(ids), K]}
    meta["phylo_neighbours"] = ("patristic distance (substitutions per site) on "
                                + os.path.basename(TREE))
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=1)
    refresh_manifest(WEB)

    first = [struct.unpack("<e", dist[i * 2:i * 2 + 2])[0] for i in range(0, len(dist) // 2, K)]
    overlap = [record["nbr_overlap"] for record in otus]
    print(f"  nearest relative: median distance "
          f"{sorted(first)[len(first) // 2]:.4f}")
    print(f"  ecological neighbours that are also phylogenetic ones, of {K}: "
          f"mean {sum(overlap) / len(overlap):.1f}")


if __name__ == "__main__":
    main()
