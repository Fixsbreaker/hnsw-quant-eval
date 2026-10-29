#!/usr/bin/env python3
"""Converts an ann-benchmarks HDF5 file to the TEXMEX vecs format.

Usage: hdf5_to_vecs.py sift-128-euclidean.hdf5 data/sift
Writes sift_base.fvecs, sift_query.fvecs and sift_groundtruth.ivecs.
Needs: pip install h5py numpy
"""
import os
import sys

import h5py
import numpy as np


def write_vecs(path, m, dtype):
    m = np.ascontiguousarray(m, dtype=dtype)
    n, d = m.shape
    out = np.empty((n, d + 1), dtype=dtype)
    out[:, 0] = np.array([d], dtype=np.int32).view(dtype)[0]
    out[:, 1:] = m
    out.tofile(path)
    print(f"{path}: {n} x {d}")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    with h5py.File(src, "r") as f:
        write_vecs(os.path.join(dst, "sift_base.fvecs"), f["train"][:], np.float32)
        write_vecs(os.path.join(dst, "sift_query.fvecs"), f["test"][:], np.float32)
        write_vecs(os.path.join(dst, "sift_groundtruth.ivecs"), f["neighbors"][:], np.int32)


if __name__ == "__main__":
    main()
