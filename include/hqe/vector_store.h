#pragma once

#include <cstddef>
#include <cstdint>

#include "hqe/io.h"

namespace hqe {

// Contiguous in-memory storage for float vectors.
// Every row starts on a 64-byte boundary (rows are padded with zeros),
// so SIMD kernels can use aligned loads.
class VectorStore {
public:
    static constexpr size_t kAlignment = 64;

    explicit VectorStore(size_t dim);
    ~VectorStore();

    VectorStore(const VectorStore&) = delete;
    VectorStore& operator=(const VectorStore&) = delete;
    VectorStore(VectorStore&& other) noexcept;
    VectorStore& operator=(VectorStore&& other) noexcept;

    // Builds a store from all rows of a matrix.
    static VectorStore from_matrix(const Matrix<float>& m);

    void reserve(size_t n);
    uint32_t add(const float* vec);  // returns id of the new vector
    void add_batch(const float* vecs, size_t n);

    const float* get(uint32_t id) const { return data_ + size_t(id) * stride_; }

    size_t dim() const { return dim_; }
    size_t stride() const { return stride_; }  // floats between rows
    size_t size() const { return size_; }
    size_t memory_bytes() const { return capacity_ * stride_ * sizeof(float); }

private:
    size_t dim_;
    size_t stride_;
    size_t size_ = 0;
    size_t capacity_ = 0;
    float* data_ = nullptr;
};

}  // namespace hqe
