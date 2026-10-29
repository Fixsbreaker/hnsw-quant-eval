#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hqe {

// Dense row-major matrix.
template <typename T>
struct Matrix {
    std::vector<T> data;
    size_t rows = 0;
    size_t dim = 0;

    T* row(size_t i) { return data.data() + i * dim; }
    const T* row(size_t i) const { return data.data() + i * dim; }
    bool empty() const { return rows == 0; }
};

// Readers for the TEXMEX formats used by SIFT1M:
// every vector is stored as int32 dim followed by dim values.
// max_rows = 0 reads the whole file.
Matrix<float> read_fvecs(const std::string& path, size_t max_rows = 0);
Matrix<int32_t> read_ivecs(const std::string& path, size_t max_rows = 0);
Matrix<float> read_bvecs(const std::string& path, size_t max_rows = 0);

void write_fvecs(const std::string& path, const Matrix<float>& m);
void write_ivecs(const std::string& path, const Matrix<int32_t>& m);

}  // namespace hqe
