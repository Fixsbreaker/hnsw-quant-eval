#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hqe/io.h"
#include "hqe/vector_store.h"

namespace hqe {

struct Neighbor {
    uint32_t id;
    float dist;
};

// Exact k-nearest-neighbour search by full scan over the store (L2).
// Used as the reference answer for the graph index.
class BruteForceIndex {
public:
    explicit BruteForceIndex(const VectorStore& store) : store_(store) {}

    // Returns min(k, size) neighbours sorted by distance, ties broken by id.
    std::vector<Neighbor> search(const float* query, size_t k) const;

    // Searches every row of queries. Result has queries.rows rows of k ids,
    // -1 where fewer than k vectors exist. threads = 0 uses all cores.
    Matrix<int32_t> search_batch(const Matrix<float>& queries, size_t k, unsigned threads = 0) const;

private:
    const VectorStore& store_;
};

}  // namespace hqe
