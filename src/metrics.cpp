#include "hqe/metrics.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace hqe {

double recall_at_k(const Matrix<int32_t>& result, const Matrix<int32_t>& groundtruth, size_t k) {
    if (result.rows != groundtruth.rows) throw std::invalid_argument("row count mismatch");
    if (k == 0 || k > result.dim || k > groundtruth.dim)
        throw std::invalid_argument("k is larger than the number of columns");
    if (result.rows == 0) return 0.0;

    size_t hits = 0;
    std::vector<int32_t> truth(k);
    for (size_t q = 0; q < result.rows; ++q) {
        std::copy_n(groundtruth.row(q), k, truth.begin());
        std::sort(truth.begin(), truth.end());
        const int32_t* got = result.row(q);
        for (size_t j = 0; j < k; ++j)
            if (got[j] >= 0 && std::binary_search(truth.begin(), truth.end(), got[j])) ++hits;
    }
    return double(hits) / double(result.rows * k);
}

}  // namespace hqe
