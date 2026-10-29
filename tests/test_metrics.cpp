#include <cmath>

#include "check.h"
#include "hqe/metrics.h"

using namespace hqe;

static Matrix<int32_t> ids(size_t rows, size_t dim, std::vector<int32_t> data) {
    Matrix<int32_t> m;
    m.rows = rows;
    m.dim = dim;
    m.data = std::move(data);
    return m;
}

int main() {
    const auto gt = ids(2, 3, {1, 2, 3, 4, 5, 6});

    CHECK(recall_at_k(gt, gt, 3) == 1.0);
    // order inside top-k does not matter
    CHECK(recall_at_k(ids(2, 3, {3, 1, 2, 6, 5, 4}), gt, 3) == 1.0);
    // 2 of 3 found in first query, 0 of 3 in the second
    CHECK(std::fabs(recall_at_k(ids(2, 3, {1, 2, 9, 7, 8, 9}), gt, 3) - 2.0 / 6.0) < 1e-12);
    // recall@1 only looks at the first column
    CHECK(recall_at_k(ids(2, 3, {1, 0, 0, 4, 0, 0}), gt, 1) == 1.0);
    // missing results (-1) never count
    CHECK(recall_at_k(ids(2, 3, {-1, -1, -1, -1, -1, -1}), gt, 3) == 0.0);

    CHECK_THROWS(recall_at_k(gt, gt, 4));
    CHECK_THROWS(recall_at_k(ids(1, 3, {1, 2, 3}), gt, 3));

    std::puts("test_metrics: ok");
    return 0;
}
