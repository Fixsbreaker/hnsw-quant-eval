#include "hqe/dataset.h"

#include <filesystem>
#include <stdexcept>

namespace hqe {

Dataset load_sift1m(const std::string& dir, size_t max_base, size_t max_query) {
    namespace fs = std::filesystem;
    const fs::path root(dir);
    const fs::path base_path = root / "sift_base.fvecs";
    const fs::path query_path = root / "sift_query.fvecs";
    const fs::path gt_path = root / "sift_groundtruth.ivecs";

    if (!fs::exists(base_path) || !fs::exists(query_path))
        throw std::runtime_error("SIFT1M not found in " + dir +
                                 " (run scripts/download_sift1m.sh)");

    Dataset ds;
    ds.base = read_fvecs(base_path.string(), max_base);
    ds.query = read_fvecs(query_path.string(), max_query);
    if (ds.base.dim != ds.query.dim)
        throw std::runtime_error("base and query dimensions differ");

    const size_t full_base =
        fs::file_size(base_path) / (sizeof(int32_t) + ds.base.dim * sizeof(float));
    if (ds.base.rows == full_base && fs::exists(gt_path)) {
        ds.groundtruth = read_ivecs(gt_path.string(), max_query);
        if (ds.groundtruth.rows != ds.query.rows)
            throw std::runtime_error("ground truth does not match queries");
    }
    return ds;
}

}  // namespace hqe
