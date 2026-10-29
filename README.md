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
./scripts/download_sift1m.sh          # downloads into data/sift (about 160 MB)
./build/sift_info data/sift           # sizes, memory, load time
```

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
