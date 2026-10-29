#pragma once

#include <cstddef>
#include <cstdint>

#include "hqe/io.h"

namespace hqe {

// recall@k: share of the true k nearest neighbours that appear among the first k
// returned ids, averaged over all queries. Both matrices need at least k columns
// and the same number of rows.
double recall_at_k(const Matrix<int32_t>& result, const Matrix<int32_t>& groundtruth, size_t k);

}  // namespace hqe
