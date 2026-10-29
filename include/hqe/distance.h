#pragma once

#include <cstddef>

namespace hqe {

// Squared Euclidean distance. Scalar version, SIMD kernels come later.
inline float l2_sqr(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

}  // namespace hqe
