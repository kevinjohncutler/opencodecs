"""Independent slide metadata supports selective frame placement."""
from __future__ import annotations

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from opencodecs._dicom import DicomError, DicomFile
from opencodecs._dicom_pyramid import DicomWsiPyramid
from opencodecs.core.io import BufferDataSource


def write_slide(path, *, undefined=False, implicit=False, compressed=False,
                multiple_paths=False, multiple_planes=False, duplicate=False, big_endian=False):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import (ExplicitVRLittleEndian, ImplicitVRLittleEndian,
                             VLWholeSlideMicroscopyImageStorage, RLELossless, ExplicitVRBigEndian, generate_uid)
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = VLWholeSlideMicroscopyImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = (ExplicitVRBigEndian if big_endian else
                                    ImplicitVRLittleEndian if implicit else ExplicitVRLittleEndian)
    ds.SOPClassUID = VLWholeSlideMicroscopyImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.Modality = "SM"
    ds.Rows, ds.Columns = 128, 192
    ds.NumberOfFrames = 6
    ds.TotalPixelMatrixRows, ds.TotalPixelMatrixColumns = 241, 563
    ds.NumberOfOpticalPaths = 1
    ds.TotalPixelMatrixFocalPlanes = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    origins = [(128, 192), (0, 384), (128, 0), (0, 0), (128, 384), (0, 192)]
    if duplicate:
        origins[-1] = origins[0]
    frames = []
    groups = []
    for index, (row, column) in enumerate(origins):
        frame = (np.arange(128 * 192, dtype="<u2").reshape(128, 192) + index * 101).astype("<u2")
        frames.append(frame)
        group, position, optical = Dataset(), Dataset(), Dataset()
        position.RowPositionInTotalImagePixelMatrix = row + 1
        position.ColumnPositionInTotalImagePixelMatrix = column + 1
        position.ZOffsetInSlideCoordinateSystem = float(index if multiple_planes else 0)
        optical.OpticalPathIdentifier = str(index if multiple_paths else 1)
        group.PlanePositionSlideSequence = Sequence([position])
        group.OpticalPathIdentificationSequence = Sequence([optical])
        if undefined:
            group.is_undefined_length_sequence_item = True
            position.is_undefined_length_sequence_item = True
            group["PlanePositionSlideSequence"].is_undefined_length = True
        groups.append(group)
    ds.PerFrameFunctionalGroupsSequence = Sequence(groups)
    if undefined:
        ds["PerFrameFunctionalGroupsSequence"].is_undefined_length = True
    ds.PixelData = np.stack(frames).astype(">u2" if big_endian else "<u2").tobytes()
    if compressed:
        ds.compress(RLELossless)
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    reference = pydicom.dcmread(path).pixel_array
    expected = np.zeros((241, 563), dtype="u2")
    for tile, (row, column) in zip(reference, origins):
        height, width = min(128, 241-row), min(192, 563-column)
        expected[row:row+height, column:column+width] = tile[:height, :width]
    return expected, origins


@pytest.mark.parametrize("undefined", [False, True])
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_slide_region_reads_only_intersecting_frames(tmp_path, undefined, implicit, compressed):
    path = tmp_path / "slide.dcm"
    expected, origins = write_slide(path, undefined=undefined, implicit=implicit, compressed=compressed)
    class Counting(BufferDataSource):
        def __init__(self, data):
            super().__init__(data)
            self.calls = []
        def read_at(self, offset, size):
            self.calls.append((offset, size))
            return super().read_at(offset, size)
    source = Counting(path.read_bytes())
    file = DicomFile(source)
    with DicomWsiPyramid([file]) as reader:
        assert file.slide_frame_positions == tuple(origins)
        source.calls.clear()
        actual = reader.read_region(0, y=(137, 177), x=(207, 267))
        np.testing.assert_array_equal(actual, expected[137:177, 207:267])
        assert sum(size for _, size in source.calls) < len(path.read_bytes()) // 2
        actual = reader.read_region(0, y=(111, 241), x=(177, 563))
        np.testing.assert_array_equal(actual, expected[111:241, 177:563])


@pytest.mark.parametrize("option", ["multiple_paths", "multiple_planes", "duplicate"])
def test_slide_rejects_ambiguous_placement(tmp_path, option):
    path = tmp_path / "ambiguous.dcm"
    write_slide(path, **{option: True})
    with DicomWsiPyramid(path) as reader:
        with pytest.raises(DicomError, match="optical paths|focal planes|overlapping"):
            reader.read_region(0, y=(137, 177), x=(207, 267))


def test_slide_missing_positions_fails_before_decoding(tmp_path):
    path = tmp_path / "missing.dcm"
    write_slide(path)
    ds = pydicom.dcmread(path)
    del ds.PerFrameFunctionalGroupsSequence
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    with DicomWsiPyramid(path) as reader:
        with pytest.raises(DicomError, match="Per-Frame Functional Groups"):
            reader.read_region(0, y=(0, 50), x=(0, 50))


def test_slide_big_endian_positions_and_pixels(tmp_path):
    path = tmp_path / "big.dcm"
    expected, _ = write_slide(path, big_endian=True, undefined=True)
    with DicomWsiPyramid(path) as reader:
        got = reader.read_region(0, y=(100, 150), x=(180, 300))
        np.testing.assert_array_equal(got, expected[100:150, 180:300])


@pytest.mark.parametrize("keyword", ["NumberOfOpticalPaths", "TotalPixelMatrixFocalPlanes"])
def test_slide_declared_multiple_axes_rejected(tmp_path, keyword):
    path = tmp_path / "axes.dcm"
    write_slide(path)
    ds = pydicom.dcmread(path)
    setattr(ds, keyword, 2)
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    with DicomWsiPyramid(path) as reader:
        with pytest.raises(DicomError, match="optical path and focal plane"):
            reader.read_region(0, y=(0, 30), x=(0, 30))
