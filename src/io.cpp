#include "hqe/io.h"

#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace hqe {

namespace {

using FilePtr = std::unique_ptr<std::FILE, int (*)(std::FILE*)>;

FilePtr open_file(const std::string& path, const char* mode) {
    FilePtr f(std::fopen(path.c_str(), mode), &std::fclose);
    if (!f) throw std::runtime_error("cannot open file: " + path);
    return f;
}

// E is the element type on disk, T is the element type in memory.
template <typename E, typename T>
Matrix<T> read_vecs(const std::string& path, size_t max_rows) {
    FilePtr f = open_file(path, "rb");

    int32_t dim = 0;
    if (std::fread(&dim, sizeof(dim), 1, f.get()) != 1)
        throw std::runtime_error("empty file: " + path);
    if (dim <= 0 || dim > (1 << 20))
        throw std::runtime_error("bad dimension in " + path);
    std::rewind(f.get());

    const size_t file_size = std::filesystem::file_size(path);
    const size_t record = sizeof(int32_t) + size_t(dim) * sizeof(E);
    if (file_size % record != 0)
        throw std::runtime_error("file size is not a multiple of record size: " + path);

    size_t rows = file_size / record;
    if (max_rows > 0 && max_rows < rows) rows = max_rows;

    Matrix<T> m;
    m.rows = rows;
    m.dim = size_t(dim);
    m.data.resize(rows * m.dim);

    std::vector<E> buf;
    if constexpr (!std::is_same_v<E, T>) buf.resize(m.dim);

    for (size_t i = 0; i < rows; ++i) {
        int32_t d = 0;
        if (std::fread(&d, sizeof(d), 1, f.get()) != 1 || d != dim)
            throw std::runtime_error("broken record " + std::to_string(i) + " in " + path);
        if constexpr (std::is_same_v<E, T>) {
            if (std::fread(m.row(i), sizeof(E), m.dim, f.get()) != m.dim)
                throw std::runtime_error("unexpected end of file: " + path);
        } else {
            if (std::fread(buf.data(), sizeof(E), m.dim, f.get()) != m.dim)
                throw std::runtime_error("unexpected end of file: " + path);
            T* out = m.row(i);
            for (size_t j = 0; j < m.dim; ++j) out[j] = static_cast<T>(buf[j]);
        }
    }
    return m;
}

template <typename T>
void write_vecs(const std::string& path, const Matrix<T>& m) {
    FilePtr f = open_file(path, "wb");
    const int32_t dim = static_cast<int32_t>(m.dim);
    for (size_t i = 0; i < m.rows; ++i) {
        if (std::fwrite(&dim, sizeof(dim), 1, f.get()) != 1 ||
            std::fwrite(m.row(i), sizeof(T), m.dim, f.get()) != m.dim)
            throw std::runtime_error("write failed: " + path);
    }
}

}  // namespace

Matrix<float> read_fvecs(const std::string& path, size_t max_rows) {
    return read_vecs<float, float>(path, max_rows);
}

Matrix<int32_t> read_ivecs(const std::string& path, size_t max_rows) {
    return read_vecs<int32_t, int32_t>(path, max_rows);
}

Matrix<float> read_bvecs(const std::string& path, size_t max_rows) {
    return read_vecs<uint8_t, float>(path, max_rows);
}

void write_fvecs(const std::string& path, const Matrix<float>& m) { write_vecs(path, m); }

void write_ivecs(const std::string& path, const Matrix<int32_t>& m) { write_vecs(path, m); }

}  // namespace hqe
