#include "hqe/vector_store.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace hqe {

namespace {

constexpr size_t kFloatsPerLine = VectorStore::kAlignment / sizeof(float);

size_t round_up(size_t n, size_t m) { return (n + m - 1) / m * m; }

}  // namespace

VectorStore::VectorStore(size_t dim) : dim_(dim), stride_(round_up(dim, kFloatsPerLine)) {
    if (dim == 0) throw std::invalid_argument("dimension must be positive");
}

VectorStore::~VectorStore() { std::free(data_); }

VectorStore::VectorStore(VectorStore&& other) noexcept
    : dim_(other.dim_),
      stride_(other.stride_),
      size_(std::exchange(other.size_, 0)),
      capacity_(std::exchange(other.capacity_, 0)),
      data_(std::exchange(other.data_, nullptr)) {}

VectorStore& VectorStore::operator=(VectorStore&& other) noexcept {
    if (this != &other) {
        std::free(data_);
        dim_ = other.dim_;
        stride_ = other.stride_;
        size_ = std::exchange(other.size_, 0);
        capacity_ = std::exchange(other.capacity_, 0);
        data_ = std::exchange(other.data_, nullptr);
    }
    return *this;
}

VectorStore VectorStore::from_matrix(const Matrix<float>& m) {
    VectorStore store(m.dim);
    store.add_batch(m.data.data(), m.rows);
    return store;
}

void VectorStore::reserve(size_t n) {
    if (n <= capacity_) return;
    // Byte size is a multiple of 64 because stride_ is, as aligned_alloc requires.
    float* fresh = static_cast<float*>(std::aligned_alloc(kAlignment, n * stride_ * sizeof(float)));
    if (!fresh) throw std::bad_alloc();
    if (data_) std::memcpy(fresh, data_, size_ * stride_ * sizeof(float));
    std::free(data_);
    data_ = fresh;
    capacity_ = n;
}

uint32_t VectorStore::add(const float* vec) {
    if (size_ >= std::numeric_limits<uint32_t>::max())
        throw std::length_error("too many vectors");
    if (size_ == capacity_) reserve(std::max<size_t>(16, capacity_ * 2));
    float* row = data_ + size_ * stride_;
    std::memcpy(row, vec, dim_ * sizeof(float));
    std::fill(row + dim_, row + stride_, 0.0f);
    return static_cast<uint32_t>(size_++);
}

void VectorStore::add_batch(const float* vecs, size_t n) {
    reserve(size_ + n);
    for (size_t i = 0; i < n; ++i) add(vecs + i * dim_);
}

}  // namespace hqe
