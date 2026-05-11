#!/bin/bash
# Download the opencodecs reference test corpus into .test_data/.
#
# These are publicly-available files from microscope vendors and
# scientific repositories that test cases use as ground truth. The
# .test_data/ directory is gitignored — files are local only, and the
# tests that need them are pytest-skipif-gated on file presence.
#
# Run from the repo root:
#   bash tests/download_test_corpus.sh
#
# Total size: ~730 MB.

set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

mkdir -p .test_data/czi .test_data/ome_tiff .test_data/ome_zarr

# Download helper that skips if the file already exists with non-zero size.
fetch() {
    local url="$1"
    local dest="$2"
    if [ -s "$dest" ]; then
        echo "[skip] $dest already present"
        return 0
    fi
    echo "[fetch] $url -> $dest"
    curl -L --fail --max-time 600 -o "$dest.tmp" "$url"
    mv "$dest.tmp" "$dest"
}

# ----- CZI: pyramid CZI from OME public mirror -----
fetch \
    "https://downloads.openmicroscopy.org/images/Zeiss-CZI/zenodo-10577186/2023_11_30__RecognizedCode-27.czi" \
    ".test_data/czi/ome_axioscan_pyramid.czi"

# ----- OME-TIFF: sub-resolution pyramid (bioformats output) -----
fetch \
    "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/sub-resolutions/Z-stack/retina_large.ome.tiff" \
    ".test_data/ome_tiff/retina_pyramid.ome.tiff"

# ----- OME-Zarr: subset of an IDR dataset (NGFF v0.4) -----
ZARR=".test_data/ome_zarr/idr_sample.zarr"
ZARR_URL="https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr"
mkdir -p "$ZARR/2"
fetch "$ZARR_URL/.zattrs" "$ZARR/.zattrs"
fetch "$ZARR_URL/.zgroup" "$ZARR/.zgroup"
fetch "$ZARR_URL/2/.zarray" "$ZARR/2/.zarray"
# Sample a handful of chunks across c=0,1 / z=0,100,200,235 / y=0 / x=0
for c in 0 1; do
    for z in 0 100 200 235; do
        mkdir -p "$ZARR/2/$c/$z/0"
        fetch "$ZARR_URL/2/$c/$z/0/0" "$ZARR/2/$c/$z/0/0"
    done
done

echo
echo "Corpus ready:"
du -sh .test_data/czi/* .test_data/ome_tiff/* .test_data/ome_zarr/* 2>/dev/null
