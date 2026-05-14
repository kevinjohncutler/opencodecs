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
# Total size with all corpora: ~750 MB.
# Pass --light to skip the big microscopy files (CZI + OME-TIFF) and
# only fetch the small format-coverage corpora (~25 MB total) — useful
# in CI and for users who just want to run codec correctness tests.

set -eu

LIGHT=0
if [ "${1:-}" = "--light" ]; then
    LIGHT=1
    echo "[light mode] skipping large microscopy fixtures (CZI + OME-TIFF)"
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

mkdir -p \
    .test_data/czi .test_data/ome_tiff .test_data/ome_zarr \
    .test_data/png/pngsuite .test_data/png/kodak24 \
    .test_data/dicom .test_data/fits .test_data/heif .test_data/lerc

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
if [ "$LIGHT" = "0" ]; then
fetch \
    "https://downloads.openmicroscopy.org/images/Zeiss-CZI/zenodo-10577186/2023_11_30__RecognizedCode-27.czi" \
    ".test_data/czi/ome_axioscan_pyramid.czi"

# ----- OME-TIFF: sub-resolution pyramid (bioformats output) -----
fetch \
    "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/sub-resolutions/Z-stack/retina_large.ome.tiff" \
    ".test_data/ome_tiff/retina_pyramid.ome.tiff"
fi  # not LIGHT

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

# ----- PngSuite: canonical PNG variant test set (~70 KB tarball) -----
# Covers every PNG bit depth, color type, filter, gamma, transparency,
# ICC, and interlacing case. Single source of truth for PNG decoder
# correctness across the spec surface.
PNGSUITE_TARBALL=".test_data/png/PngSuite-2017jul19.tgz"
fetch \
    "http://www.schaik.com/pngsuite/PngSuite-2017jul19.tgz" \
    "$PNGSUITE_TARBALL"
if [ ! -f ".test_data/png/pngsuite/basn0g08.png" ]; then
    echo "[extract] $PNGSUITE_TARBALL -> .test_data/png/pngsuite/"
    tar -xzf "$PNGSUITE_TARBALL" -C .test_data/png/pngsuite/ 2>/dev/null || \
        tar -xzf "$PNGSUITE_TARBALL" -C .test_data/png/pngsuite/ --include='*.png'
fi

# ----- Kodak24: 24-image photographic codec benchmark set (~10 MB) -----
# THE standard image-codec quality bench set. Same images used in every
# JPEG / WebP / AVIF / JXL paper for the last 30 years. Lets us measure
# lossy-codec quality + speed on real photos, not random data.
for i in $(seq -w 1 24); do
    fetch \
        "https://r0k.us/graphics/kodak/kodak/kodim${i}.png" \
        ".test_data/png/kodak24/kodim${i}.png"
done

# ----- DICOM: pydicom-data emri_small (functional MRI volume, ~85 KB) -----
# Public-domain DICOM sample. Exercises our _dicomweb path on real
# bytes from a different toolchain than ours.
fetch \
    "https://raw.githubusercontent.com/pydicom/pydicom-data/master/data_store/data/emri_small.dcm" \
    ".test_data/dicom/emri_small.dcm"

# ----- FITS: HorseHead nebula (~1.6 MB) -----
# Astropy tutorials sample. Real Hubble exposure as int16 — exercises
# our _fits reader on data from a non-microscopy scientific domain.
fetch \
    "https://www.astropy.org/astropy-data/tutorials/FITS-images/HorseHead.fits" \
    ".test_data/fits/HorseHead.fits"

# ----- HEIF: Nokia conformance file C001 (~870 KB) -----
# Reference HEIC from the HEIF conformance suite. Real Apple/Nokia
# HEIC layout (single still). License: BSD-3 (Nokia conformance repo).
fetch \
    "https://raw.githubusercontent.com/nokiatech/heif_conformance/master/conformance_files/C001.heic" \
    ".test_data/heif/C001.heic"

# ----- LERC: ESRI's reference test data (~230 KB total) -----
# Two files from the LERC repo: a float32 raster (DEM elevation) and a
# byte RGB raster (satellite image). Cross-validates our _lerc against
# the canonical encoder.
fetch \
    "https://raw.githubusercontent.com/Esri/lerc/master/testData/california_400_400_1_float.lerc2" \
    ".test_data/lerc/california_float.lerc2"
fetch \
    "https://raw.githubusercontent.com/Esri/lerc/master/testData/bluemarble_256_256_3_byte.lerc2" \
    ".test_data/lerc/bluemarble_byte.lerc2"

echo
echo "Corpus ready:"
du -sh .test_data/*/ 2>/dev/null | sort -k 2
