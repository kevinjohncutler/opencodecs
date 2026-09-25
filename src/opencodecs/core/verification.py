"""Optional segment round-trip verification with bounded comparison scratch.

This checks codec payloads. Container indexes, metadata, destination writes,
and subsequent storage corruption require separate file-level validation.
"""

from __future__ import annotations

import operator

import numpy as np

_COMPARE_BYTES = 64 * 1024


class LosslessVerificationError(ValueError):
    """A decoded segment does not preserve its original representation."""


def _fail(reason, codec):
    label = "segment" if codec is None else f"segment encoded with {codec!r}"
    raise LosslessVerificationError(f"Lossless verification failed for {label}: {reason}")


def _byte_view(data, codec):
    try:
        view = memoryview(data)
        if not view.c_contiguous:
            _fail("byte payload must be C-contiguous", codec)
        return view.cast("B")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, LosslessVerificationError):
            raise
        _fail("payload does not expose a contiguous byte buffer", codec)


def _compare_bytes(left, right, codec, label="bytes"):
    if len(left) != len(right):
        _fail(f"byte length differs: {len(left)} versus {len(right)}", codec)
    for start in range(0, len(left), _COMPARE_BYTES):
        if (left[start:start + _COMPARE_BYTES].tobytes() !=
                right[start:start + _COMPARE_BYTES].tobytes()):
            _fail(f"{label} differ in byte block starting at {start}", codec)


def _effective_byteorder(dtype):
    if dtype.byteorder == "=":
        return "<" if np.little_endian else ">"
    return dtype.byteorder


def assert_bit_exact(original, decoded, *, codec=None):
    """Compare exact bytes or image shape, scalar type, and pixel bits.

    Arrays compare in logical C order after byte-order normalization. No
    floating-point conversion occurs, so NaN payloads and signed zeros matter.
    The caller must keep the original stable until this function returns.
    Neither input is mutated or retained. Comparison copies stay bounded.
    """
    if isinstance(original, np.ndarray):
        if not isinstance(decoded, np.ndarray):
            _fail("decoded image is not an array", codec)
        if original.shape != decoded.shape:
            _fail(f"shape differs: {original.shape} versus {decoded.shape}", codec)
        if (original.dtype.hasobject or decoded.dtype.hasobject or
                original.dtype.fields is not None or decoded.dtype.fields is not None or
                original.dtype.itemsize > _COMPARE_BYTES or
                decoded.dtype.itemsize > _COMPARE_BYTES):
            _fail("object, structured, and oversized image dtypes are unsupported", codec)
        if (original.dtype.kind, original.dtype.itemsize) != (
                decoded.dtype.kind, decoded.dtype.itemsize):
            _fail(f"dtype differs: {original.dtype} versus {decoded.dtype}", codec)
        if not original.nbytes:
            return
        if (original.flags.c_contiguous and decoded.flags.c_contiguous and
                _effective_byteorder(original.dtype) == _effective_byteorder(decoded.dtype)):
            # Identical storage orders need no per-element copies or swapping.
            # Keep only bounded byte blocks, never an image-sized comparison.
            _compare_bytes(_byte_view(original, codec), _byte_view(decoded, codec),
                           codec, label="pixel bits")
            return
        step = max(1, _COMPARE_BYTES // max(1, original.dtype.itemsize))
        for start in range(0, original.size, step):
            # flat slicing copies only this block, including strided arrays.
            left = original.flat[start:start + step]
            right = decoded.flat[start:start + step]
            for block in (left, right):
                if block.dtype.byteorder == ">" or (
                        block.dtype.byteorder == "=" and not np.little_endian):
                    block.byteswap(inplace=True)
            if left.tobytes() != right.tobytes():
                _fail(f"pixel bits differ in element block starting at {start}", codec)
        return
    if isinstance(decoded, np.ndarray):
        _fail("decoded byte payload unexpectedly became an image", codec)
    left, right = _byte_view(original, codec), _byte_view(decoded, codec)
    _compare_bytes(left, right, codec)


def verification_reservation(original):
    """Additional decoded bytes and conservative bounded comparison scratch.

    Accept an input buffer/array or its nonnegative byte count. This excludes
    retained input, compressed output, and native decoder working memory.
    Writers must include those costs separately in their task reservations.
    """
    if isinstance(original, np.ndarray):
        size = original.nbytes
    elif isinstance(original, (int, np.integer)):
        size = operator.index(original)
    else:
        size = memoryview(original).nbytes
    if size < 0:
        raise ValueError("verification byte count must be nonnegative")
    return size + 4 * min(size, _COMPARE_BYTES)


def verify_segment(original, encoded, codec, decode_options=None):
    """Decode through the shared dispatcher and require a bit-exact result.

    Image-codec inputs retain shape and scalar type. Byte-codec array inputs
    are compared as their encoded C-contiguous byte representation. Decoder
    output is independently allocated; caller destinations are prohibited.
    """
    from .segment_compression import decode_segment, segment_input_kind

    options = {} if decode_options is None else dict(decode_options)
    if "out" in options:
        raise ValueError("verification owns its decoded output; do not supply out")
    expected = original
    if segment_input_kind(codec) == "bytes" and isinstance(original, np.ndarray):
        expected = _byte_view(original, codec)
    verify_with_decoder(expected, encoded,
                        lambda payload: decode_segment(payload, codec, **options),
                        codec=codec)


def verify_with_decoder(original, encoded, decoder, *, codec=None):
    """Verify using an existing container decoder, including its transforms.

    ``decoder(encoded)`` must return independently owned bytes or pixels and
    reverse every transform applied before compression. This helper permits
    container-specific codec namespaces without duplicating their decoders.
    """
    try:
        decoded = decoder(encoded)
    except Exception as exc:
        raise LosslessVerificationError(
            f"Lossless verification failed for {codec!r}: decode failed: {exc}"
        ) from exc
    assert_bit_exact(original, decoded, codec=codec)


class DeferredVerification:
    """One verification task overlapped with the caller's next encode.

    The producer thread owns this object. It must finish the previous task
    before submitting another, and budget the retained previous segment,
    decoded verification result, and concurrently encoded next segment.
    A single worker and native worker budget prevent nested decoder pools.
    """

    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor
        from .pipeline import WorkerBudget
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="opencodecs-verify")
        self._budget = WorkerBudget(1)
        self._pending = None
        self._closed = False

    def submit(self, check, *args, **kwargs):
        """Start one check; its arguments must remain stable until finish."""
        from functools import partial
        if self._closed:
            raise RuntimeError("verification worker is closed")
        if self._pending is not None:
            raise RuntimeError("finish the pending verification before submitting")
        task = partial(check, *args, **kwargs)
        self._pending = self._executor.submit(self._budget.run,
                                               lambda call: call(), task)

    def finish(self):
        """Wait for the pending check, returning its result or raising failure."""
        pending = self._pending
        if pending is None:
            return None
        try:
            return pending.result()
        finally:
            # An interrupt while waiting must not make an active task appear
            # finished or permit a second submission with retained buffers.
            if pending.done():
                self._pending = None

    def close(self):
        """Join the worker and propagate any unobserved verification failure."""
        if not self._closed:
            self._closed = True
            try:
                self.finish()
            finally:
                self._executor.shutdown(wait=True)

    def __enter__(self):
        if self._closed:
            raise RuntimeError("verification worker is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception:
                # Join outstanding work without replacing the caller's error.
                pass


__all__ = ["LosslessVerificationError", "assert_bit_exact", "verify_segment",
           "verify_with_decoder", "verification_reservation", "DeferredVerification"]
