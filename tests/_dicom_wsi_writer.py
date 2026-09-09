"""Write a VL Whole Slide Microscopy DICOM series with pydicom.

The corpus has no whole-slide DICOM, and this package cannot write one:
its DICOM support is read-only. pydicom is the reference implementation
of the format, so letting it build the fixture keeps the writer outside
the code under test -- and it means the tags are spelled the way
pydicom spells them rather than the way this reader happens to expect.

Returns None when pydicom is unavailable, so callers skip.
"""

from __future__ import annotations

import numpy as np


def write_wsi_series(directory, levels, *, series_uid=None,
                     imaged_volume_mm=(1.0, 1.0), samples_per_pixel=1):
    """Write one instance per entry of ``levels`` (2-D uint8 arrays).

    Files are named so that alphabetical order is the REVERSE of
    resolution order, which is what proves the reader orders levels by
    extent rather than by filename.
    """
    try:
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import (VLWholeSlideMicroscopyImageStorage,
                                 ExplicitVRLittleEndian, generate_uid)
    except ImportError:
        return None

    series_uid = series_uid or generate_uid()
    paths = []
    n = len(levels)
    for i, img in enumerate(levels):
        img = np.ascontiguousarray(img, dtype=np.uint8)
        rows, cols = img.shape[:2]
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = VLWholeSlideMicroscopyImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = VLWholeSlideMicroscopyImageStorage
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = series_uid
        ds.Modality = "SM"
        ds.ImageType = (["ORIGINAL", "PRIMARY", "VOLUME", "NONE"] if i == 0
                        else ["DERIVED", "PRIMARY", "VOLUME", "RESAMPLED"])
        # The tags that make an instance a pyramid LEVEL: the extent it
        # covers. Rows/Columns are one tile on a tiled slide.
        ds.TotalPixelMatrixRows = rows
        ds.TotalPixelMatrixColumns = cols
        ds.Rows = rows
        ds.Columns = cols
        ds.NumberOfFrames = 1
        ds.SamplesPerPixel = samples_per_pixel
        ds.PhotometricInterpretation = ("MONOCHROME2" if samples_per_pixel == 1
                                        else "RGB")
        if samples_per_pixel > 1:
            ds.PlanarConfiguration = 0
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.ImagedVolumeHeight = float(imaged_volume_mm[0])
        ds.ImagedVolumeWidth = float(imaged_volume_mm[1])
        ds.ImagedVolumeDepth = 0.01
        ds.PixelData = img.tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        # Reverse-alphabetical against resolution.
        path = directory / f"inst_{n - 1 - i:02d}.dcm"
        ds.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths
