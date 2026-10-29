#include <filesystem>
#include <string>

#include "check.h"
#include "hqe/dataset.h"

using namespace hqe;

static Matrix<float> make(size_t rows, size_t dim, float shift) {
    Matrix<float> m;
    m.rows = rows;
    m.dim = dim;
    for (size_t i = 0; i < rows * dim; ++i) m.data.push_back(float(i) + shift);
    return m;
}

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "hqe_test_sift";
    fs::create_directories(dir);

    write_fvecs((dir / "sift_base.fvecs").string(), make(20, 4, 0));
    write_fvecs((dir / "sift_query.fvecs").string(), make(3, 4, 0.5f));
    Matrix<int32_t> gt;
    gt.rows = 3;
    gt.dim = 2;
    gt.data = {0, 1, 1, 2, 2, 3};
    write_ivecs((dir / "sift_groundtruth.ivecs").string(), gt);

    Dataset full = load_sift1m(dir.string());
    CHECK(full.base.rows == 20 && full.query.rows == 3);
    CHECK(full.groundtruth.rows == 3);

    // Ground truth refers to the full base, so it must not be loaded for a subset.
    Dataset part = load_sift1m(dir.string(), 10, 2);
    CHECK(part.base.rows == 10 && part.query.rows == 2);
    CHECK(part.groundtruth.empty());

    CHECK_THROWS(load_sift1m((dir / "missing").string()));

    fs::remove_all(dir);
    std::puts("test_dataset: ok");
    return 0;
}
