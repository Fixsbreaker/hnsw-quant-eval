// Exact search on SIFT1M (or a prefix of the base) and accuracy check.
//
// With the full base the result is compared with the official ground truth.
// With a subset (default 10 000 vectors) the official ground truth does not
// apply, so the exact answers are saved and later used to evaluate the graph.
//
// Usage: bf_eval [--data DIR] [--base N] [--queries N] [--k K]
//                [--threads T] [--gt FILE] [--out FILE]
//   --base 0     use the full base
//   --gt FILE    compare with this ground truth instead of the official one
//   --out FILE   where to save the computed answers

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "hqe/brute_force.h"
#include "hqe/dataset.h"
#include "hqe/metrics.h"
#include "hqe/vector_store.h"

namespace {

struct Options {
    std::string data = "data/sift";
    size_t base = 10000;
    size_t queries = 0;
    size_t k = 100;
    unsigned threads = 0;
    std::string gt;
    std::string out;
};

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: bf_eval [--data DIR] [--base N] [--queries N] [--k K]\n"
                 "               [--threads T] [--gt FILE] [--out FILE]\n");
    std::exit(2);
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (i + 1 >= argc) usage();
        const char* v = argv[++i];
        if (!std::strcmp(a, "--data")) o.data = v;
        else if (!std::strcmp(a, "--base")) o.base = std::strtoull(v, nullptr, 10);
        else if (!std::strcmp(a, "--queries")) o.queries = std::strtoull(v, nullptr, 10);
        else if (!std::strcmp(a, "--k")) o.k = std::strtoull(v, nullptr, 10);
        else if (!std::strcmp(a, "--threads")) o.threads = unsigned(std::strtoul(v, nullptr, 10));
        else if (!std::strcmp(a, "--gt")) o.gt = v;
        else if (!std::strcmp(a, "--out")) o.out = v;
        else usage();
    }
    if (o.k == 0) usage();
    return o;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void print_recall(const hqe::Matrix<int32_t>& result, const hqe::Matrix<int32_t>& gt) {
    for (size_t k : {1, 10, 100}) {
        if (k > result.dim || k > gt.dim) break;
        std::printf("recall@%-3zu %.4f\n", k, hqe::recall_at_k(result, gt, k));
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    try {
        auto t0 = std::chrono::steady_clock::now();
        hqe::Dataset ds = hqe::load_sift1m(opt.data, opt.base, opt.queries);
        std::printf("loaded %zu base, %zu queries, dim %zu in %.2f s\n", ds.base.rows,
                    ds.query.rows, ds.base.dim, seconds_since(t0));

        hqe::VectorStore store = hqe::VectorStore::from_matrix(ds.base);
        ds.base = {};  // the store keeps its own copy
        std::printf("store: %zu vectors, %.1f MB\n", store.size(), store.memory_bytes() / 1e6);

        hqe::BruteForceIndex index(store);
        t0 = std::chrono::steady_clock::now();
        hqe::Matrix<int32_t> result = index.search_batch(ds.query, opt.k, opt.threads);
        const double sec = seconds_since(t0);
        std::printf("brute force k=%zu: %.2f s, %.1f queries/s\n", opt.k, sec,
                    ds.query.rows / sec);

        hqe::Matrix<int32_t> gt;
        if (!opt.gt.empty()) {
            gt = hqe::read_ivecs(opt.gt, ds.query.rows);
        } else {
            gt = std::move(ds.groundtruth);
        }
        if (!gt.empty()) {
            if (gt.rows != result.rows)
                throw std::runtime_error("ground truth has fewer rows than queries");
            print_recall(result, gt);
        } else {
            std::printf("no ground truth for this base size, nothing to compare\n");
        }

        std::string out = opt.out;
        if (out.empty() && opt.base != 0)
            out = opt.data + "/gt_base" + std::to_string(store.size()) + "_k" +
                  std::to_string(opt.k) + ".ivecs";
        if (!out.empty()) {
            hqe::write_ivecs(out, result);
            std::printf("saved exact answers to %s\n", out.c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
