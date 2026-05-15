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
    .test_data/dicom .test_data/fits .test_data/heif .test_data/lerc \
    .test_data/tiff/libtiff_pics .test_data/tiff/cog \
    .test_data/tiff/geotiff .test_data/tiff/wsi \
    .test_data/ndtiff .test_data/nd2 .test_data/lif .test_data/oib \
    .test_data/oir .test_data/vsi

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

# ----- libtiff pics-3.8.0: canonical TIFF compatibility corpus (~3.5 MB) -----
# Covers compression variants (NONE, PACKBITS, LZW, DEFLATE, CCITT,
# OLD-JPEG), bit depths (1/4/8), color types (gray, RGB, palette),
# strip vs tile layout, and multi-page. The de-facto reference set for
# any TIFF reader.
PICS_TARBALL=".test_data/tiff/pics-3.8.0.tar.gz"
fetch \
    "https://download.osgeo.org/libtiff/pics-3.8.0.tar.gz" \
    "$PICS_TARBALL"
if [ ! -f ".test_data/tiff/libtiff_pics/cramps.tif" ]; then
    echo "[extract] $PICS_TARBALL -> .test_data/tiff/libtiff_pics/"
    tar -xzf "$PICS_TARBALL" -C .test_data/tiff/libtiff_pics/ \
        --strip-components=1
fi

# ----- COG: real Cloud-Optimized GeoTIFFs from rio-cogeo's test corpus -----
# These are byte-exact COG layouts (overviews-first then header+IFD,
# tiled, sized to be tile-aligned). The 2000px file has full pyramid;
# image_float and image_colormap exercise the dtype + colormap paths.
fetch \
    "https://raw.githubusercontent.com/cogeotiff/rio-cogeo/master/tests/fixtures/image_2000px.tif" \
    ".test_data/tiff/cog/image_2000px.tif"
fetch \
    "https://raw.githubusercontent.com/cogeotiff/rio-cogeo/master/tests/fixtures/image_float.tif" \
    ".test_data/tiff/cog/image_float.tif"
fetch \
    "https://raw.githubusercontent.com/cogeotiff/rio-cogeo/master/tests/fixtures/image_colormap.tif" \
    ".test_data/tiff/cog/image_colormap.tif"

# ----- GeoTIFF: ECW Cylindrical Equal-Area + USGS DEM (~5.5 MB) -----
# cea.tif is tile-based with full GeoKeys; i30dem.tif is a strip-based
# elevation model. Together they exercise both layout types as
# emitted by GDAL.
fetch \
    "https://download.osgeo.org/geotiff/samples/gdal_eg/cea.tif" \
    ".test_data/tiff/geotiff/cea.tif"
fetch \
    "https://download.osgeo.org/geotiff/samples/usgs/i30dem.tif" \
    ".test_data/tiff/geotiff/i30dem.tif"

# ----- Aperio WSI: CMU-1-Small-Region.svs (~1.9 MB) -----
# Real pathology whole-slide image (SVS = TIFF with Aperio extensions).
# Multi-page pyramid + JPEG-compressed tiles. The smallest public
# Aperio sample.
fetch \
    "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs" \
    ".test_data/tiff/wsi/CMU-1-Small-Region.svs"

# ----- Small CZI: idr0011 plate well (~43 MB) -----
# Different ZEN version + much smaller content than the 505 MB
# Axioscan pyramid. Useful for fast CI runs that just need to
# exercise the CZI parser on real bytes.
fetch \
    "https://downloads.openmicroscopy.org/images/Zeiss-CZI/idr0011/Plate1-Blue-A_TS-Stinger/Plate1-Blue-A-02-Scene-1-P2-E1-01.czi" \
    ".test_data/czi/idr0011_plate1_scene1.czi"

# ----- NDTiff v3: Micro-Manager test sample (~1.5 MB) -----
# Populates the previously-empty .test_data/ndtiff/ subdir. Real
# Micro-Manager 2 NDTiff layout: index file + multi-position TIFF
# stack. Exercises our _ndtiff reader.
fetch \
    "https://raw.githubusercontent.com/micro-manager/NDStorage/main/test_data/v3/ndtiff3.2_monochrome/NDTiff.index" \
    ".test_data/ndtiff/NDTiff.index"
fetch \
    "https://raw.githubusercontent.com/micro-manager/NDStorage/main/test_data/v3/ndtiff3.2_monochrome/NDTiff3.2_monochrome_NDTiffStack.tif" \
    ".test_data/ndtiff/NDTiff3.2_monochrome_NDTiffStack.tif"

# ----- ND2 sample: Nikon NIS-Elements (~13 MB) -----
# Smallest ND2 in the OME mirror. Exercises our (future) ND2 reader
# via the nd2 Python package. For now: format-detection sentinel.
fetch \
    "https://downloads.openmicroscopy.org/images/ND2/aryeh/MeOh_high_fluo_007.nd2" \
    ".test_data/nd2/MeOh_high_fluo_007.nd2"

# ----- LIF sample: Leica LAS-X (~230 KB) -----
# Tiny, multi-image Leica LIF from the OME mirror. The smallest
# real LIF I could find. Exercises our (future) LIF reader.
fetch \
    "https://downloads.openmicroscopy.org/images/Leica-LIF/michael/PR2729_frameOrderCombinedScanTypes.lif" \
    ".test_data/lif/PR2729_frameOrderCombinedScanTypes.lif"

# ----- OIB sample: Olympus FluoView (~25 MB) -----
# Real Olympus FluoView OIB. Exercises our (future) OIB reader.
fetch \
    "https://downloads.openmicroscopy.org/images/Olympus-FluoView/imagesc-71616/20220824_4492_cord_dapi__iba568_60x.oib" \
    ".test_data/oib/imagesc_71616_60x.oib"

# ----- VSI: Olympus CellSens virtual-slide index (~110 KB) -----
# Real Olympus VSI from the OME mirror. The top-level .vsi file is
# a TIFF (II*\0 magic) containing the thumbnail + metadata; full-res
# pixel data lives in a sibling ".../_NAME_/stackN/frame_t_N.ets"
# tree. We fetch the index AND the .ets companion so the VSI tests
# can exercise the SIS/ETS partial parser too.
fetch \
    "https://downloads.openmicroscopy.org/images/CellSens/zenodo-17590655/metadataTest_01.vsi" \
    ".test_data/vsi/metadataTest_01.vsi"
mkdir -p .test_data/vsi/_metadataTest_01_/stack1
fetch \
    "https://downloads.openmicroscopy.org/images/CellSens/zenodo-17590655/_metadataTest_01_/stack1/frame_t_0.ets" \
    ".test_data/vsi/_metadataTest_01_/stack1/frame_t_0.ets"

# ----- OIR sample: Olympus FluoView newer format (~25 MB) -----
# Real Olympus OIR from the OME mirror. OIR uses an undocumented
# OLYMPUSRAWFORMAT binary container; we don't have a native parser
# yet, but the corpus file lets us add format-detection tests and
# eventually build one.
fetch \
    "https://downloads.openmicroscopy.org/images/Olympus-OIR/etienne/amy%20slice%20z%20stack_0001.oir" \
    ".test_data/oir/amy_slice_z_stack.oir"

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
