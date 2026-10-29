#include "hqe/brute_force.h"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <thread>

#include "hqe/distance.h"

namespace hqe {

namespace {

bool closer(const Neighbor& a, const Neighbor& b) {
    return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
}

struct Closer {
    bool operator()(const Neighbor& a, const Neighbor& b) const { return closer(a, b); }
};

}  // namespace

std::vector<Neighbor> BruteForceIndex::search(const float* query, size_t k) const {
    if (k == 0) return {};
    // Max-heap on distance: the top is the worst of the current best k.
    std::priority_queue<Neighbor, std::vector<Neighbor>, Closer> heap;
    const size_t n = store_.size();
    const size_t dim = store_.dim();
    for (size_t i = 0; i < n; ++i) {
        const Neighbor cand{static_cast<uint32_t>(i), l2_sqr(query, store_.get(uint32_t(i)), dim)};
        if (heap.size() < k) {
            heap.push(cand);
        } else if (closer(cand, heap.top())) {
            heap.pop();
            heap.push(cand);
        }
    }
    std::vector<Neighbor> out(heap.size());
    for (size_t i = out.size(); i-- > 0;) {
        out[i] = heap.top();
        heap.pop();
    }
    return out;
}

Matrix<int32_t> BruteForceIndex::search_batch(const Matrix<float>& queries, size_t k,
                                              unsigned threads) const {
    if (queries.dim != store_.dim()) throw std::invalid_argument("query dimension mismatch");

    Matrix<int32_t> result;
    result.rows = queries.rows;
    result.dim = k;
    result.data.assign(queries.rows * k, -1);

    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    threads = static_cast<unsigned>(std::min<size_t>(threads, std::max<size_t>(1, queries.rows)));

    auto worker = [&](size_t begin, size_t end) {
        for (size_t q = begin; q < end; ++q) {
            const std::vector<Neighbor> nn = search(queries.row(q), k);
            int32_t* out = result.row(q);
            for (size_t j = 0; j < nn.size(); ++j) out[j] = static_cast<int32_t>(nn[j].id);
        }
    };

    std::vector<std::thread> pool;
    const size_t chunk = (queries.rows + threads - 1) / threads;
    for (unsigned t = 0; t < threads; ++t) {
        const size_t begin = t * chunk;
        const size_t end = std::min(queries.rows, begin + chunk);
        if (begin >= end) break;
        pool.emplace_back(worker, begin, end);
    }
    for (auto& th : pool) th.join();
    return result;
}

}  // namespace hqe
