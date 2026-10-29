#!/usr/bin/env bash
# Downloads SIFT1M into data/sift.
#
# Default source is the ann-benchmarks mirror (HDF5, fast), converted to
# fvecs/ivecs with hdf5_to_vecs.py (needs h5py and numpy).
# SOURCE=ftp uses the original TEXMEX archive, which is often very slow.
set -euo pipefail

DATA_DIR="${1:-data}"
SOURCE="${SOURCE:-mirror}"
FTP_URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
MIRROR_URL="http://ann-benchmarks.com/sift-128-euclidean.hdf5"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$DATA_DIR"
if [ -f "$DATA_DIR/sift/sift_base.fvecs" ]; then
    echo "SIFT1M already in $DATA_DIR/sift"
    exit 0
fi

if [ "$SOURCE" = "ftp" ]; then
    echo "Downloading $FTP_URL"
    curl -fL --retry 3 -o "$DATA_DIR/sift.tar.gz" "$FTP_URL"
    tar -xzf "$DATA_DIR/sift.tar.gz" -C "$DATA_DIR"
    rm "$DATA_DIR/sift.tar.gz"
else
    HDF5="$DATA_DIR/sift-128-euclidean.hdf5"
    if [ ! -f "$HDF5" ]; then
        echo "Downloading $MIRROR_URL"
        curl -fL --retry 3 -o "$HDF5" "$MIRROR_URL"
    fi
    "${PYTHON:-python3}" "$SCRIPT_DIR/hdf5_to_vecs.py" "$HDF5" "$DATA_DIR/sift"
    rm "$HDF5"
fi
ls -lh "$DATA_DIR/sift"
