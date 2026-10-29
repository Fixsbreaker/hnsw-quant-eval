#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "check.h"
#include "hqe/brute_force.h"
#include "hqe/distance.h"

using namespace hqe;

// Independent reference: full sort of all distances in double precision.
static std::vector<uint32_t> naive_knn(const Matrix<float>& base, const float* q, size_t k) {
    std::vector<double> dist(base.rows);
    for (size_t i = 0; i < base.rows; ++i) {
        double s = 0;
        for (size_t j = 0; j < base.dim; ++j) {
            const double d = double(base.row(i)[j]) - q[j];
            s += d * d;
        }
        dist[i] = s;
    }
    std::vector<uint32_t> ids(base.rows);
    std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) {
        return dist[a] < dist[b] || (dist[a] == dist[b] && a < b);
    });
    ids.resize(std::min(k, ids.size()));
    return ids;
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    Matrix<float> base;
    base.rows = 2000;
    base.dim = 32;
    for (size_t i = 0; i < base.rows * base.dim; ++i) base.data.push_back(uni(rng));

    Matrix<float> queries;
    queries.rows = 50;
    queries.dim = 32;
    for (size_t i = 0; i < queries.rows * queries.dim; ++i) queries.data.push_back(uni(rng));

    VectorStore store = VectorStore::from_matrix(base);
    BruteForceIndex index(store);

    const size_t k = 10;
    for (size_t q = 0; q < queries.rows; ++q) {
        const auto got = index.search(queries.row(q), k);
        const auto want = naive_knn(base, queries.row(q), k);
        CHECK(got.size() == k);
        for (size_t j = 0; j < k; ++j) CHECK(got[j].id == want[j]);
        for (size_t j = 1; j < k; ++j) CHECK(got[j - 1].dist <= got[j].dist);
    }

    // Batch search must match single search, with any number of threads.
    for (unsigned threads : {1u, 3u, 0u}) {
        Matrix<int32_t> res = index.search_batch(queries, k, threads);
        CHECK(res.rows == queries.rows && res.dim == k);
        for (size_t q = 0; q < queries.rows; ++q) {
            const auto single = index.search(queries.row(q), k);
            for (size_t j = 0; j < k; ++j) CHECK(res.row(q)[j] == int32_t(single[j].id));
        }
    }

    // A query equal to a base vector finds itself first.
    CHECK(index.search(base.row(123), 1)[0].id == 123);

    // k larger than the store.
    VectorStore tiny(2);
    const float p[2] = {0, 0};
    tiny.add(p);
    BruteForceIndex small(tiny);
    CHECK(small.search(p, 5).size() == 1);

    std::puts("test_brute_force: ok");
    return 0;
}
