# hnsw-quant-eval

Vector search based on HNSW and Product Quantization, with an evaluation
on SIFT1M and text embeddings (AI/RAG use case).

## Build

Requirements: C++17 compiler, CMake 3.16+.

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

## SIFT1M

```bash
pip install h5py numpy
./scripts/download_sift1m.sh          # downloads into data/sift
./build/sift_info data/sift           # sizes, memory, load time
```

By default the data comes from the ann-benchmarks mirror (HDF5, 525 MB) and is
converted to fvecs/ivecs by `scripts/hdf5_to_vecs.py`. `SOURCE=ftp` uses the
original TEXMEX archive, which is usually very slow.

## Exact search and recall

```bash
# full base: compares brute force with the official ground truth
./build/bf_eval --base 0 --queries 1000

# 10k subset: computes exact top-100 and saves data/sift/gt_base10000_k100.ivecs
./build/bf_eval --base 10000

# check any result file against that ground truth
./build/bf_eval --base 10000 --gt data/sift/gt_base10000_k100.ivecs
```

The saved file is the reference answer for evaluating the graph index on 10k vectors.

### Results (Apple M1 Pro, 10 threads, scalar L2)

| Base | Queries | k | Time | Queries/s | recall@1 | recall@10 | recall@100 |
|---|---|---|---|---|---|---|---|
| 1 000 000 | 1 000 | 100 | 9.2 s | 109 | 0.9950 | 0.9991 | 0.9999 |
| 10 000 | 10 000 | 100 | 1.1 s | 9 380 | 1.0000 | 1.0000 | 1.0000 |

Recall on the full base is below 1 only because of ties: SIFT coordinates are
integers, so some vectors are at exactly the same distance and the official
ground truth picks a different id. All 5 top-1 mismatches in the first 1000
queries are such ties, and the top-10 distances match the ground truth for all
1000 queries.

## Layout

```
include/hqe/   public headers
  io.h            fvecs / ivecs / bvecs readers and writers
  dataset.h       SIFT1M loader
  vector_store.h  aligned in-memory vector storage
  distance.h      distance functions
  brute_force.h   exact k-NN search
  metrics.h       recall@k
src/           implementation
tools/         command line programs
tests/         unit tests (ctest)
scripts/       dataset download
```
