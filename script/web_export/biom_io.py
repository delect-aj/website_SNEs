"""Minimal BIOM 2.1 reader built on h5py.

The `biom-format` package needs a C extension that does not build on this
machine, and the export pipeline only ever reads one file format. The HDF5
layout BIOM 2.1 writes is small enough to read directly:

    observation/ids                 (n_features,)  feature ids, stored sorted
    observation/matrix/{data,indices,indptr}       feature-major CSR
    sample/ids                      (n_samples,)   sample ids
    sample/matrix/{data,indices,indptr}            sample-major CSR

Both matrices describe the same table, so only one is read.
"""

import h5py
import numpy as np
from scipy.sparse import csr_matrix


def read_biom(path):
    """Read a BIOM 2.1 table into a feature-major sparse matrix.

    Parameters
    ----------
    path : str
        Path to the ``.biom`` file.

    Returns
    -------
    feature_ids : numpy.ndarray
        Feature (OTU) identifiers of shape ``(n_features,)``.
    sample_ids : numpy.ndarray
        Sample identifiers of shape ``(n_samples,)``.
    table : scipy.sparse.csr_matrix
        Counts of shape ``(n_features, n_samples)``, the same orientation and
        values as ``biom.load_table(...).matrix_data``.

    Raises
    ------
    ValueError
        If the stored shape disagrees with the id arrays, which would mean the
        two halves of the file were written by different tool versions.
    """
    with h5py.File(path, "r") as handle:
        feature_ids = _strings(handle["observation/ids"])
        sample_ids = _strings(handle["sample/ids"])
        group = handle["observation/matrix"]
        data, indices, indptr = (group[k][()] for k in ("data", "indices", "indptr"))
        shape = tuple(int(v) for v in handle.attrs["shape"])

    if shape != (len(feature_ids), len(sample_ids)):
        raise ValueError(
            f"{path}: stored shape {shape} disagrees with "
            f"{len(feature_ids)} features x {len(sample_ids)} samples")

    # The compressed axis of this group is the observations, so the block is
    # the CSR form of the (features, samples) matrix.
    table = csr_matrix((data, indices, indptr), shape=shape)
    return feature_ids, sample_ids, table


def read_metadata(path, keys):
    """Read selected columns of a sample metadata TSV, indexed by sample id.

    Parameters
    ----------
    path : str
        Tab-separated file with a ``sample`` column.
    keys : sequence of str
        Columns to return.

    Returns
    -------
    dict
        One ``{column: {sample_id: value}}`` mapping per requested key. Values
        are returned as written, i.e. as strings.
    """
    columns = {key: {} for key in keys}
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index_of = {name: header.index(name) for name in ["sample"] + list(keys)}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            sample = fields[index_of["sample"]]
            for key in keys:
                columns[key][sample] = fields[index_of[key]]
    return columns


def _strings(dataset):
    """Decode an HDF5 array of bytes into an array of ``str``."""
    return np.array([value.decode() if isinstance(value, bytes) else str(value)
                     for value in dataset[()]], dtype=object)
