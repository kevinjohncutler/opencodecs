"""Every reader takes the same kinds of source.

opencodecs advertises readers that accept "a path, bytes, a file-like,
an http(s) URL, or a read_at callable". That was true of some of them.
Which ones came down to an accident of plumbing: readers built on
``core._io_helpers.open_read_at`` took buffers and file objects, while
readers built on ``core.io.coerce_data_source`` took only a path or a
DataSource -- so OIR refused bytes outright, and ND2, LIF, OIB and CZI
worked around it by writing the buffer to a temporary file to get a
path back. A 40 MB CZI already in memory went to disk to be read.

These tests pin the uniform surface rather than the plumbing, so it
does not matter which helper a reader uses tomorrow.
"""

from __future__ import annotations

import io
import mmap
import pathlib
import tempfile

import numpy as np
import pytest

import opencodecs as oc

CORPUS = pathlib.Path(__file__).resolve().parent.parent / ".test_data"

# Every way a caller might hand a reader the same file.
SOURCE_KINDS = ["path", "str", "bytes", "bytearray", "memoryview",
                "mmap", "BytesIO", "open file"]


def _make_source(kind, path, stack):
    raw = path.read_bytes()
    if kind == "path":
        return pathlib.Path(path)
    if kind == "str":
        return str(path)
    if kind == "bytes":
        return raw
    if kind == "bytearray":
        return bytearray(raw)
    if kind == "memoryview":
        return memoryview(raw)
    if kind == "mmap":
        fh = open(path, "rb")
        stack.append(fh)
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        stack.append(mm)
        return mm
    if kind == "BytesIO":
        return io.BytesIO(raw)
    if kind == "open file":
        fh = open(path, "rb")
        stack.append(fh)
        return fh
    raise AssertionError(kind)


@pytest.fixture(scope="module")
def volume():
    return (np.sin(np.arange(8 * 32 * 32) / 11.0)
            .reshape(8, 32, 32) * 100).astype(np.int16)


@pytest.fixture(scope="module")
def image():
    return (np.sin(np.arange(64 * 64) / 7.0)
            .reshape(64, 64) * 100 + 128).astype(np.uint8)


@pytest.fixture(scope="module")
def samples(tmp_path_factory, volume, image):
    """One file per reader: written where we can, corpus where we cannot."""
    tmp = tmp_path_factory.mktemp("srckinds")
    out = {}

    def _try(name, fn):
        try:
            p = fn()
        except Exception:                                     # noqa: BLE001
            return
        if p is not None and pathlib.Path(p).is_file():
            out[name] = pathlib.Path(p)

    def _write(name, arr, **kw):
        ext = (oc.get_codec(name).file_extensions or [f".{name}"])[0]
        p = tmp / f"{name}{ext}"
        oc.get_codec(name).encode(arr, dest=str(p), **kw)
        return p

    _try("tiff", lambda: _write("tiff", image))
    _try("jxl", lambda: _write("jxl", image))
    _try("gif", lambda: _write("gif", image))
    _try("mrc", lambda: (oc.write_mrc(str(tmp / "s.mrc"), volume),
                         tmp / "s.mrc")[1])
    _try("nifti", lambda: (oc.write_nifti(str(tmp / "s.nii"), volume),
                           tmp / "s.nii")[1])

    def _h5():
        import h5py
        p = tmp / "s.h5"
        with h5py.File(p, "w") as f:
            f["data"] = volume
        return p
    _try("hdf5", _h5)

    if CORPUS.is_dir():
        for name in ("czi", "dicom", "dm", "eer", "emd", "fits", "lif",
                     "nd2", "nrrd", "oib", "oir", "vsi"):
            exts = oc.get_codec(name).file_extensions or ()
            for sub in sorted(CORPUS.iterdir()):
                if not sub.is_dir() or name in out:
                    continue
                for f in sorted(sub.iterdir()):
                    # Skip macOS AppleDouble sidecars: they carry the
                    # extension of the file they shadow and are not it.
                    if (f.is_file() and not f.name.startswith("._")
                            and f.stat().st_size > 1024
                            and any(f.name.lower().endswith(e)
                                    for e in exts)):
                        out[name] = f
                        break
    return out


def _reader_names():
    from opencodecs.core.codec import Codec
    names = []
    for info in sorted(oc.list_codecs(), key=lambda c: c["name"]):
        cls = type(oc.get_codec(info["name"]))
        for k in cls.__mro__:
            if "open" in k.__dict__:
                if k is not Codec:
                    names.append(info["name"])
                break
    return names


@pytest.mark.parametrize("kind", SOURCE_KINDS)
@pytest.mark.parametrize("name", _reader_names())
def test_reader_accepts_every_source_kind(name, kind, samples):
    """A reader that takes a path must take the same bytes any other way.

    Skipped rather than failed when no sample exists, because "we have
    no file for this format" and "this reader rejects buffers" are
    different facts and only one of them is a bug.
    """
    if name not in samples:
        pytest.skip(f"no sample file available for {name}")
    stack = []
    try:
        src = _make_source(kind, samples[name], stack)
        with oc.get_codec(name).open(src) as r:
            assert r.shape is not None
            assert r.dtype is not None
    finally:
        for obj in reversed(stack):
            try:
                obj.close()
            except Exception:                                 # noqa: BLE001
                pass


def test_a_file_like_source_is_not_consumed_twice():
    """LIF read the caller's stream once per branch.

    The native parser drained it, then the readlif fallback read it
    again from EOF, got b"", wrote an empty temp file and failed with
    an error about a closed handle that pointed nowhere near the
    cause. Any reader that tries more than one backend can regrow this,
    so the check is on the stream, not on LIF.
    """
    class CountingStream(io.BytesIO):
        def __init__(self, data):
            super().__init__(data)
            self.reads_from_start = 0

        def read(self, *a, **kw):
            if self.tell() == 0:
                self.reads_from_start += 1
            return super().read(*a, **kw)

    payload = b"not a real image file, but it is bytes"
    s = CountingStream(payload)
    try:
        oc.get_codec("lif").open(s)
    except Exception:                                         # noqa: BLE001
        pass
    assert s.reads_from_start <= 1, (
        f"the source was read from the start {s.reads_from_start} times; "
        f"a second pass sees an already-drained stream")


def test_buffer_data_source_addresses_bytes():
    """BufferDataSource slices by BYTE offset whatever it wraps.

    A memoryview of an int32 array reports itemsize 4, so slicing it
    with byte offsets would silently read the wrong region -- the kind
    of wrong that returns plausible data.
    """
    from opencodecs.core.io import BufferDataSource

    raw = bytes(range(256))
    for buf in (raw, bytearray(raw), memoryview(raw),
                np.frombuffer(raw, dtype=np.uint8),
                np.frombuffer(raw, dtype=np.int32)):
        ds = BufferDataSource(buf)
        assert ds.size == 256, type(buf)
        assert ds.read_at(0, 4) == raw[:4], type(buf)
        assert ds.read_at(10, 5) == raw[10:15], type(buf)
        assert ds.read_at(250, 99) == raw[250:], "short tail read"
        assert ds.read_at(300, 4) == b"", "read past the end"


def test_buffer_data_source_rejects_negative_reads():
    from opencodecs.core.io import BufferDataSource
    ds = BufferDataSource(b"0123456789")
    with pytest.raises(ValueError):
        ds.read_at(-1, 4)
    with pytest.raises(ValueError):
        ds.read_at(0, -4)


def test_coerce_data_source_accepts_what_readers_accept():
    """The helper the vendor parsers share, checked directly."""
    from opencodecs.core.io import coerce_data_source, DataSource

    raw = bytes(range(256))
    for src in (raw, bytearray(raw), memoryview(raw), io.BytesIO(raw)):
        ds, owns, size = coerce_data_source(src)
        assert isinstance(ds, DataSource), type(src)
        assert size == 256, type(src)
        assert ds.read_at(4, 4) == raw[4:8], type(src)


def test_coerce_data_source_leaves_the_callers_handle_open(tmp_path):
    """Closing something we were merely handed is not ours to do."""
    from opencodecs.core.io import coerce_data_source

    p = tmp_path / "x.bin"
    p.write_bytes(bytes(range(256)))
    with p.open("rb") as fh:
        fh.seek(10)
        ds, owns, size = coerce_data_source(fh)
        assert size == 256
        assert ds.read_at(0, 4) == bytes(range(4))
        assert not fh.closed, "the caller's file was closed"
        assert fh.tell() == 10, "the caller's stream position moved"


def test_coerce_data_source_still_rejects_nonsense():
    from opencodecs.core.io import coerce_data_source
    with pytest.raises(TypeError, match="unsupported source"):
        coerce_data_source(object())
    with pytest.raises(TypeError, match="unsupported source"):
        coerce_data_source(42)
