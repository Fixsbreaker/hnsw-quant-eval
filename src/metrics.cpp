#include "hqe/metrics.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "hqe/distance.h"

namespace hqe {

namespace {

void check_shapes(const Matrix<int32_t>& result, const Matrix<int32_t>& groundtruth, size_t k) {
    if (result.rows != groundtruth.rows) throw std::invalid_argument("row count mismatch");
    if (k == 0 || k > result.dim || k > groundtruth.dim)
        throw std::invalid_argument("k is larger than the number of columns");
}

}  // namespace

double recall_at_k(const Matrix<int32_t>& result, const Matrix<int32_t>& groundtruth, size_t k) {
    check_shapes(result, groundtruth, k);
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

double recall_at_k_ties(const Matrix<int32_t>& result, const Matrix<int32_t>& groundtruth,
                        const VectorStore& store, const Matrix<float>& queries, size_t k,
                        double rel_eps) {
    check_shapes(result, groundtruth, k);
    if (queries.rows != result.rows) throw std::invalid_argument("query count mismatch");
    if (queries.dim != store.dim()) throw std::invalid_argument("query dimension mismatch");
    if (result.rows == 0) return 0.0;

    auto valid = [&](int32_t id) { return id >= 0 && size_t(id) < store.size(); };
    const size_t dim = store.dim();
    size_t hits = 0;
    std::vector<int32_t> seen(k);
    for (size_t q = 0; q < result.rows; ++q) {
        const float* query = queries.row(q);
        const int32_t kth = groundtruth.row(q)[k - 1];
        if (!valid(kth)) throw std::out_of_range("ground truth id outside the store");
        const double threshold = double(l2_sqr(query, store.get(uint32_t(kth)), dim)) * (1 + rel_eps);

        // A duplicated id must not be counted twice.
        std::copy_n(result.row(q), k, seen.begin());
        std::sort(seen.begin(), seen.end());
        const auto end = std::unique(seen.begin(), seen.end());
        for (auto it = seen.begin(); it != end; ++it)
            if (valid(*it) && l2_sqr(query, store.get(uint32_t(*it)), dim) <= threshold) ++hits;
    }
    return double(hits) / double(result.rows * k);
}

}  // namespace hqe
