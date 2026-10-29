#!/usr/bin/env bash
# Downloads SIFT1M (ANN_SIFT1M from the TEXMEX corpus) into data/sift.
set -euo pipefail

DATA_DIR="${1:-data}"
URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

mkdir -p "$DATA_DIR"
if [ -f "$DATA_DIR/sift/sift_base.fvecs" ]; then
    echo "SIFT1M already in $DATA_DIR/sift"
    exit 0
fi

echo "Downloading $URL"
curl -fL --retry 3 -o "$DATA_DIR/sift.tar.gz" "$URL"
tar -xzf "$DATA_DIR/sift.tar.gz" -C "$DATA_DIR"
rm "$DATA_DIR/sift.tar.gz"
ls -lh "$DATA_DIR/sift"
