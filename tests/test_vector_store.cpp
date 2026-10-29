#include <cstdint>
#include <utility>
#include <vector>

#include "check.h"
#include "hqe/distance.h"
#include "hqe/vector_store.h"

using namespace hqe;

int main() {
    // dim 3 is padded to one 64-byte line
    VectorStore s(3);
    CHECK(s.stride() == 16);
    CHECK(s.size() == 0);

    for (int i = 0; i < 100; ++i) {
        const float v[3] = {float(i), float(i) + 1, float(i) + 2};
        CHECK(s.add(v) == uint32_t(i));
    }
    CHECK(s.size() == 100);
    for (uint32_t i = 0; i < 100; ++i) {
        const float* row = s.get(i);
        CHECK(reinterpret_cast<uintptr_t>(row) % VectorStore::kAlignment == 0);
        CHECK(row[0] == float(i) && row[2] == float(i) + 2);
        CHECK(row[3] == 0.0f && row[15] == 0.0f);  // padding
    }

    Matrix<float> m;
    m.rows = 2;
    m.dim = 130;
    m.data.assign(260, 1.0f);
    VectorStore t = VectorStore::from_matrix(m);
    CHECK(t.size() == 2 && t.stride() == 144);

    VectorStore moved = std::move(t);
    CHECK(moved.size() == 2 && t.size() == 0);

    const float a[4] = {1, 2, 3, 4};
    const float b[4] = {2, 2, 1, 0};
    CHECK(l2_sqr(a, b, 4) == 1 + 0 + 4 + 16);

    std::puts("test_vector_store: ok");
    return 0;
}
