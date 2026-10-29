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
