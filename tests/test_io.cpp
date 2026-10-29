#include <cstdio>
#include <filesystem>
#include <string>

#include "check.h"
#include "hqe/io.h"

using namespace hqe;

int main() {
    const auto dir = std::filesystem::temp_directory_path();
    const std::string fpath = (dir / "hqe_test.fvecs").string();
    const std::string ipath = (dir / "hqe_test.ivecs").string();

    Matrix<float> m;
    m.rows = 5;
    m.dim = 3;
    for (size_t i = 0; i < m.rows * m.dim; ++i) m.data.push_back(float(i) * 0.5f);
    write_fvecs(fpath, m);

    Matrix<float> r = read_fvecs(fpath);
    CHECK(r.rows == 5 && r.dim == 3);
    CHECK(r.data == m.data);

    Matrix<float> part = read_fvecs(fpath, 2);
    CHECK(part.rows == 2);
    CHECK(part.row(1)[2] == m.row(1)[2]);

    Matrix<int32_t> g;
    g.rows = 2;
    g.dim = 4;
    g.data = {0, 1, 2, 3, 7, 6, 5, 4};
    write_ivecs(ipath, g);
    Matrix<int32_t> gr = read_ivecs(ipath);
    CHECK(gr.rows == 2 && gr.dim == 4 && gr.data == g.data);

    CHECK_THROWS(read_fvecs((dir / "hqe_missing.fvecs").string()));

    // Truncated file must be rejected.
    std::filesystem::resize_file(fpath, std::filesystem::file_size(fpath) - 2);
    CHECK_THROWS(read_fvecs(fpath));

    std::remove(fpath.c_str());
    std::remove(ipath.c_str());
    std::puts("test_io: ok");
    return 0;
}
