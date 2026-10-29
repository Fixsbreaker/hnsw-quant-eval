#include <cmath>
#include <utility>
#include <vector>

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

static void test_by_id() {
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
}

static void test_with_ties() {
    // 1-d points: 0, 1, -1, 2, 5. Query at 0: ids 1 and 2 are tied at distance 1.
    VectorStore store(1);
    for (float x : {0.f, 1.f, -1.f, 2.f, 5.f}) store.add(&x);
    Matrix<float> queries;
    queries.rows = 1;
    queries.dim = 1;
    queries.data = {0.f};

    const auto gt = ids(1, 2, {0, 1});
    const auto tied = ids(1, 2, {0, 2});  // other vector at the same distance
    CHECK(recall_at_k(tied, gt, 2) == 0.5);
    CHECK(recall_at_k_ties(tied, gt, store, queries, 2) == 1.0);

    // a really farther vector is still a miss
    CHECK(recall_at_k_ties(ids(1, 2, {0, 3}), gt, store, queries, 2) == 0.5);
    // duplicates count once, -1 never counts
    CHECK(recall_at_k_ties(ids(1, 2, {0, 0}), gt, store, queries, 2) == 0.5);
    CHECK(recall_at_k_ties(ids(1, 2, {-1, -1}), gt, store, queries, 2) == 0.0);
    // the same as by-id recall when there are no ties
    CHECK(recall_at_k_ties(gt, gt, store, queries, 2) == 1.0);

    CHECK_THROWS(recall_at_k_ties(gt, ids(1, 2, {0, 99}), store, queries, 2));
}

int main() {
    test_by_id();
    test_with_ties();
    std::puts("test_metrics: ok");
    return 0;
}
