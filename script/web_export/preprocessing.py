"""The Python half of the table-to-tensor contract.

`otu_attention.load_data_imdb` defines what the model sees: rank-normalize each
sample over its non-zero counts, divide by the sample's largest rank, keep the
`num_steps` most abundant OTUs, and mask padding and out-of-vocabulary
positions. The browser has to reproduce this exactly, so it is written out here
once, in a form that does not need a BIOM table in memory, and pinned by the
golden-sample test against the original.

Nothing here may drift from `otu_attention` without the golden test failing.
"""

import numpy as np

PAD_INDEX = 0
UNK_INDEX = 1


def rank_normalize(sample_counts):
    """Replace each sample's counts with ranks scaled to ``(0, 1]``.

    Mirrors ``Table.rankdata(axis='sample')`` followed by the division in
    :func:`otu_attention.read_imdb`: only non-zero counts are ranked, ties
    share the average rank, and dividing by the largest rank puts every
    non-zero entry in ``(0, 1]`` while leaving zeros at zero.

    Parameters
    ----------
    sample_counts : numpy.ndarray
        Raw counts of one sample, length ``n_features``.

    Returns
    -------
    numpy.ndarray
        Rank-normalized counts, same shape, ``float64``.
    """
    out = np.zeros_like(sample_counts, dtype=np.float64)
    nonzero = np.nonzero(sample_counts)[0]
    if nonzero.size == 0:
        return out

    values = sample_counts[nonzero].astype(np.float64)
    order = np.argsort(values, kind="mergesort")

    # Average ranks within each run of equal counts.
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    for stop in range(1, values.size + 1):
        if stop == values.size or values[order[stop]] != values[order[start]]:
            ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
            start = stop

    out[nonzero] = ranks / ranks.max()
    return out


def truncate_pad(ranked_samples, feature_indices, num_steps):
    """Turn rank-normalized rows into token, abundance and mask arrays.

    Mirrors :meth:`otu_attention.load_data_imdb.truncate_pad`. The abundance
    ordering must match ``np.argsort(row)[::-1]`` exactly, ties included, since
    a different tie order puts a different OTU in a different position and the
    attention output changes with it.

    Parameters
    ----------
    ranked_samples : numpy.ndarray
        Rank-normalized table of shape ``(n_samples, n_features)``.
    feature_indices : numpy.ndarray
        Vocabulary index of each column of `ranked_samples`, shape
        ``(n_features,)``.
    num_steps : int
        Sequence length kept per sample.

    Returns
    -------
    features : numpy.ndarray
        Vocabulary indices of shape ``(n_samples, num_steps)``, ``int64``,
        ``PAD_INDEX`` where the sample was shorter than `num_steps`.
    abundance : numpy.ndarray
        Rank-normalized abundances of shape ``(n_samples, num_steps)``,
        ``float32``, zero at padding.
    mask : numpy.ndarray
        Attention mask of shape ``(n_samples, num_steps)``, ``int64``, 0 where
        the position is padding or an OTU outside the vocabulary.
    """
    n_samples = ranked_samples.shape[0]
    features = np.full((n_samples, num_steps), PAD_INDEX, dtype=np.int64)
    abundance = np.zeros((n_samples, num_steps), dtype=np.float32)

    for i in range(n_samples):
        row = ranked_samples[i]
        nonzero = np.nonzero(row)[0]
        if nonzero.size >= num_steps:
            take = np.argsort(row)[::-1][:num_steps]
        else:
            take = nonzero
        features[i, :take.size] = feature_indices[take]
        abundance[i, :take.size] = row[take]

    mask = ((features != PAD_INDEX) & (features != UNK_INDEX)).astype(np.int64)
    return features, abundance, mask
