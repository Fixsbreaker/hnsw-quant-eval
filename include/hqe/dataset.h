#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "hqe/io.h"

namespace hqe {

struct Dataset {
    Matrix<float> base;
    Matrix<float> query;
    // Exact top-100 neighbours for every query. Only valid for the full base,
    // so it is left empty when a subset of the base is loaded.
    Matrix<int32_t> groundtruth;
};

// Loads sift_base.fvecs, sift_query.fvecs and sift_groundtruth.ivecs from dir.
// max_base / max_query = 0 means load everything.
Dataset load_sift1m(const std::string& dir, size_t max_base = 0, size_t max_query = 0);

}  // namespace hqe
