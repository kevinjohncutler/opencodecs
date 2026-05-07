"""zarr v3 BytesBytesCodec wrappers for opencodecs's compressors.

Lets users do::

    import zarr
    import opencodecs                      # registers our codecs
    from opencodecs.zarr import OcZstd, OcLz4, OcBrotli, OcBlosc2, OcDeflate

    z = zarr.create_array(
        store, shape=(...), dtype=..., chunks=...,
        codecs=[zarr.codecs.BytesCodec(), OcZstd(level=10)],
    )

These wrappers go through opencodecs's native Cython codecs rather than
``numcodecs`` (which is what zarr v3 ships by default), so the same fast
path is used whether you're calling ``opencodecs.write(format='zstd')``
or storing a chunked array via zarr.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer

try:
    from zarr.abc.codec import BytesBytesCodec
    from zarr.core.buffer.cpu import as_numpy_array_wrapper
    from zarr.core.common import JSON, parse_named_configuration
    _HAVE_ZARR = True
except ImportError:  # pragma: no cover - zarr-missing branch
    _HAVE_ZARR = False


def _make_codec(zarr_name: str, oc_name: str, doc: str):
    """Build a BytesBytesCodec subclass forwarding to opencodecs.get_codec(...)."""
    if not _HAVE_ZARR:  # pragma: no cover - zarr-missing branch
        return None

    @dataclass(frozen=True)
    class _Codec(BytesBytesCodec):  # type: ignore[misc]
        is_fixed_size = False
        level: int | None = None

        def __init__(self, *, level: int | None = None) -> None:
            object.__setattr__(self, "level", level)

        @classmethod
        def from_dict(cls, data: "dict[str, JSON]") -> "Self":
            _, cfg = parse_named_configuration(data, zarr_name)
            return cls(**cfg)  # type: ignore[arg-type]

        def to_dict(self) -> "dict[str, JSON]":
            cfg: dict = {}
            if self.level is not None:
                cfg["level"] = self.level
            return {"name": zarr_name, "configuration": cfg}

        def _encode_bytes(self, b) -> bytes:
            from . import get_codec
            # b may be a numpy uint8 array view from zarr; coerce to bytes.
            if hasattr(b, "tobytes"):
                b = bytes(b.tobytes())
            elif not isinstance(b, (bytes, bytearray, memoryview)):  # pragma: no cover - zarr always passes ndarray or buffer
                b = bytes(b)
            codec = get_codec(oc_name)
            kwargs = {} if self.level is None else {"level": self.level}
            return codec.encode(b, **kwargs)

        def _decode_bytes(self, b) -> bytes:
            from . import get_codec
            if hasattr(b, "tobytes"):
                b = bytes(b.tobytes())
            elif not isinstance(b, (bytes, bytearray, memoryview)):  # pragma: no cover - zarr always passes ndarray or buffer
                b = bytes(b)
            return get_codec(oc_name).decode(b)

        def _encode_sync(self, chunk_bytes: "Buffer",
                         chunk_spec: "ArraySpec") -> "Buffer | None":
            return as_numpy_array_wrapper(
                self._encode_bytes, chunk_bytes, chunk_spec.prototype)

        def _decode_sync(self, chunk_bytes: "Buffer",
                         chunk_spec: "ArraySpec") -> "Buffer":
            return as_numpy_array_wrapper(
                self._decode_bytes, chunk_bytes, chunk_spec.prototype)

        async def _encode_single(self, chunk_bytes, chunk_spec):
            return await asyncio.to_thread(
                self._encode_sync, chunk_bytes, chunk_spec)

        async def _decode_single(self, chunk_bytes, chunk_spec):
            return await asyncio.to_thread(
                self._decode_sync, chunk_bytes, chunk_spec)

        def compute_encoded_size(self, _input_byte_length, _chunk_spec):
            raise NotImplementedError

    _Codec.__name__ = f"Oc{zarr_name.capitalize()}Codec"
    _Codec.__doc__ = doc
    return _Codec


OcZstd = _make_codec(
    "zstd", "zstd", "zstd codec backed by opencodecs's libzstd binding")
OcLz4 = _make_codec(
    "lz4", "lz4", "LZ4 frame codec backed by opencodecs's liblz4 binding")
OcBrotli = _make_codec(
    "brotli", "brotli", "brotli codec backed by opencodecs's libbrotli binding")
OcBlosc2 = _make_codec(
    "blosc2", "blosc2", "blosc2 codec backed by opencodecs's c-blosc2 binding")
OcDeflate = _make_codec(
    "deflate", "deflate", "deflate/zlib codec backed by opencodecs")


__all__ = [
    name for name, obj in [
        ("OcZstd", OcZstd), ("OcLz4", OcLz4), ("OcBrotli", OcBrotli),
        ("OcBlosc2", OcBlosc2), ("OcDeflate", OcDeflate),
    ] if obj is not None
]
