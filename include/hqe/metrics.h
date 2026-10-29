#pragma once

#include <cstddef>
#include <cstdint>

#include "hqe/io.h"
#include "hqe/vector_store.h"

namespace hqe {

// recall@k: share of the true k nearest neighbours that appear among the first k
// returned ids, averaged over all queries. Both matrices need at least k columns
// and the same number of rows.
double recall_at_k(const Matrix<int32_t>& result, const Matrix<int32_t>& groundtruth, size_t k);

// Tie-aware recall@k (as in ann-benchmarks): a returned id counts as a hit when
// its distance to the query is not larger than the distance of the k-th true
// neighbour. Another vector at exactly the same distance is then not a miss.
// Distances are recomputed from store, which must hold the base the ground
// truth refers to. rel_eps is the relative slack on the threshold.
double recall_at_k_ties(const Matrix<int32_t>& result, const Matrix<int32_t>& groundtruth,
                        const VectorStore& store, const Matrix<float>& queries, size_t k,
                        double rel_eps = 1e-6);

}  // namespace hqe
