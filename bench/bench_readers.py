"""Benchmark the container-format readers against their reference libraries.

Each of these formats has a canonical Python implementation, so "is ours
any good" has a real answer rather than a self-comparison. Run from the
repository root after fetching the corpus:

    python bench/bench_readers.py

Measured the way docs/codec_api_conventions.md requires: warm first,
then the minimum of several runs rather than a mean, because the
minimum is the run least polluted by whatever else the machine was
doing. Reference libraries that are absent are skipped, never faked.

Read the numbers with one caveat. These are small files, so a good part
of the gap is per-call overhead in the reference libraries, several of
which build rich domain objects (hyperspy signals, nibabel images) where
we return an ndarray. That overhead is real for a caller opening
thousands of files, and it is not the same claim as being faster per
megabyte on a single large one.
"""
import time, pathlib, numpy as np

def best(fn, n=7):
    fn()
    b = float("inf")
    for _ in range(n):
        t = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t)
    return b * 1e3

rows = []

# MRC vs mrcfile
try:
    import mrcfile
    from opencodecs._mrc import MrcStream
    p = ".test_data/mrc/emd_3001.map"
    ours = best(lambda: MrcStream(p).asarray())
    theirs = best(lambda: np.asarray(mrcfile.open(p).data))
    rows.append(("mrc", "mrcfile", ours, theirs))
except Exception as e:
    rows.append(("mrc", f"skip {type(e).__name__}", 0, 0))

# NIfTI vs nibabel
try:
    import nibabel as nib
    from opencodecs._nifti import NiftiStream
    p = ".test_data/nifti/example4d.nii.gz"
    ours = best(lambda: NiftiStream(p).asarray())
    theirs = best(lambda: np.asanyarray(nib.load(p).dataobj))
    rows.append(("nifti", "nibabel", ours, theirs))
except Exception as e:
    rows.append(("nifti", f"skip {type(e).__name__}", 0, 0))

# DICOM vs pydicom
try:
    import pydicom
    from opencodecs._dicom import DicomFile
    p = ".test_data/dicom/CT_small.dcm"
    ours = best(lambda: DicomFile(p).asarray())
    theirs = best(lambda: pydicom.dcmread(p).pixel_array)
    rows.append(("dicom", "pydicom", ours, theirs))
except Exception as e:
    rows.append(("dicom", f"skip {type(e).__name__}", 0, 0))

# NRRD vs pynrrd
try:
    import nrrd as pynrrd
    from opencodecs._nrrd import NrrdFile
    p = ".test_data/nrrd/ball_gz.nrrd"
    ours = best(lambda: NrrdFile(p).asarray())
    theirs = best(lambda: pynrrd.read(p)[0])
    rows.append(("nrrd", "pynrrd", ours, theirs))
except Exception as e:
    rows.append(("nrrd", f"skip {type(e).__name__}", 0, 0))

# DM vs rosettasciio
try:
    from rsciio.digitalmicrograph import file_reader as dm_reader
    from opencodecs._dm import DmFile
    p = ".test_data/dm/test_stackbuilder_imagestack.dm3"
    ours = best(lambda: DmFile(p).asarray())
    theirs = best(lambda: dm_reader(p)[0]["data"])
    rows.append(("dm", "rosettasciio", ours, theirs))
except Exception as e:
    rows.append(("dm", f"skip {type(e).__name__}", 0, 0))

# EMD vs rosettasciio
try:
    from rsciio.emd import file_reader as emd_reader
    from opencodecs._emd import EmdFile
    p = ".test_data/emd/Si100_2x1x1_3D.emd"
    ours = best(lambda: EmdFile(p).asarray())
    theirs = best(lambda: emd_reader(p)[0]["data"])
    rows.append(("emd", "rosettasciio", ours, theirs))
except Exception as e:
    rows.append(("emd", f"skip {type(e).__name__}", 0, 0))

print(f"{'format':<8} {'reference':<14} {'ours ms':>9} {'ref ms':>9} {'speedup':>9}")
for name, ref, a, b in rows:
    if not a:
        print(f"{name:<8} {ref}")
        continue
    print(f"{name:<8} {ref:<14} {a:9.2f} {b:9.2f} {b/a:8.2f}x")
