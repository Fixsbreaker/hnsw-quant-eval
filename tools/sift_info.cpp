// Loads SIFT1M and prints basic information about it.
// Usage: sift_info [data_dir]

#include <chrono>
#include <cstdio>
#include <exception>
#include <string>

#include "hqe/dataset.h"

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "data/sift";
    try {
        const auto t0 = std::chrono::steady_clock::now();
        hqe::Dataset ds = hqe::load_sift1m(dir);
        const double sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        std::printf("base:        %zu x %zu\n", ds.base.rows, ds.base.dim);
        std::printf("query:       %zu x %zu\n", ds.query.rows, ds.query.dim);
        std::printf("groundtruth: %zu x %zu\n", ds.groundtruth.rows, ds.groundtruth.dim);
        std::printf("base memory: %.1f MB\n", ds.base.data.size() * sizeof(float) / 1e6);
        std::printf("load time:   %.2f s\n", sec);

        std::printf("base[0][0..7]:");
        for (size_t j = 0; j < 8 && j < ds.base.dim; ++j) std::printf(" %.0f", ds.base.row(0)[j]);
        std::printf("\n");
        if (!ds.groundtruth.empty())
            std::printf("nearest to query[0]: %d\n", ds.groundtruth.row(0)[0]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
