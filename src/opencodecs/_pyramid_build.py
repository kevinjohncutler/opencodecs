"""Pyramid builders — auto-downsample a full-res image into N levels.

This module is **opt-in by design**. Pyramid writers
(:func:`write_omezarr_pyramid`, :meth:`TiffWriter.write_pyramid`) take a
caller-supplied list of pre-downsampled levels, and that stays the
canonical flow. The helpers here exist for the convenience case where
you have a single full-resolution array and just want a default
pyramid written, accepting the size cost.

Default behaviour is **conservative**: levels stop being added once an
axis would drop below ``min_size`` (512 px by default) — so a 1024×1024
input produces only 2 levels (1024 + 512), not 11. Pass an explicit
``levels=N`` to override.

A pyramid roughly **doubles** the on-disk footprint (the 2× downscale
geometric series sums to 1 + 1/4 + 1/16 + ... ≈ 1.33 for 2D; up to
~2× when stored uncompressed including the original). If size is a
concern, write the single full-res array via :func:`write_zarr_array`
or pass an explicit ``levels=1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _downsample_2x_meanpool(
    arr: np.ndarray, axes: tuple[int, ...]
) -> np.ndarray:
    """Halve ``arr`` along ``axes`` via 2x2 mean pool.

    Odd-length axes drop the trailing pixel (matches OME-Zarr/Bioformats
    convention — no edge fill, no replication).
    """
    out = arr
    for ax in sorted(axes, reverse=True):
        n = out.shape[ax]
        if n < 2:
            continue
        # Trim to even length, then reshape + mean.
        if n % 2:
            sl = [slice(None)] * out.ndim
            sl[ax] = slice(0, n - 1)
            out = out[tuple(sl)]
        # Reshape last/this axis to (n/2, 2), mean along the new size-2 dim.
        new_shape = list(out.shape)
        new_shape[ax] = new_shape[ax] // 2
        new_shape.insert(ax + 1, 2)
        out_view = out.reshape(new_shape)
        # Mean as float64 then cast back to preserve dtype semantics.
        # For float inputs we stay in the input dtype (single mean pass).
        if np.issubdtype(arr.dtype, np.floating):
            out = out_view.mean(axis=ax + 1, dtype=arr.dtype)
        else:
            # Integer mean: accumulate in int32/64 then floor-divide by 2.
            # Round-half-to-even matches numpy's default float→int cast.
            acc_dtype = np.int64 if arr.dtype.itemsize >= 4 else np.int32
            out = (
                out_view.astype(acc_dtype).mean(axis=ax + 1) + 0.5
            ).astype(arr.dtype)
    return np.ascontiguousarray(out)


def make_pyramid_levels(
    image: np.ndarray,
    *,
    levels: int | None = None,
    downsample: int = 2,
    min_size: int = 512,
    axes: tuple[int, ...] | str | None = None,
) -> list[np.ndarray]:
    """Build a list of progressively-downsampled views of ``image``.

    Parameters
    ----------
    image : ndarray
        Full-resolution input. Any rank ≥ 2.
    levels : int, optional
        Total number of levels to emit (including the full-res level 0).
        ``None`` (default) auto-stops when the next level would drop any
        spatial axis below ``min_size``. Set explicitly when you want a
        guaranteed pyramid depth regardless of input size.
    downsample : int
        Downsample factor per level. Only ``2`` is currently supported.
    min_size : int
        Stop adding levels when an axis would fall below this many pixels.
        Default 512 — produces ~2-4 levels for typical microscopy / radio
        images, avoiding tiny degenerate levels that take more JSON
        metadata than pixel data.
    axes : tuple of int, or str, optional
        Which axes to downsample. If a string like ``"yx"`` or
        ``"zyx"`` is given, the trailing letters of the canonical OME
        axis order ``tczyx`` are matched to image axes (rightmost first).
        ``None`` (default) downsamples the trailing 2 spatial axes.

    Returns
    -------
    list[np.ndarray]
        ``[full_res, half, quarter, ...]``. Always has at least one
        entry (the input itself).

    Notes
    -----
    * Downsampling is **2x2 mean pool** (Bioformats/OME-Zarr default).
    * Output dtype matches input dtype. Integers use rounded division
      to avoid drift on repeated downsamples.
    * The returned arrays are independent copies — modifying one does
      not affect the others.
    """
    if downsample != 2:
        raise ValueError("only downsample=2 is currently supported")
    if image.ndim < 2:
        raise ValueError(
            f"make_pyramid_levels: image must be at least 2D; got {image.ndim}"
        )
    if axes is None:
        # Default: trailing 2 axes (the YX plane in canonical order).
        ax_tuple: tuple[int, ...] = (image.ndim - 2, image.ndim - 1)
    elif isinstance(axes, str):
        # Match "yx", "zyx", etc. against trailing letters of "tczyx".
        canonical = "tczyx"
        if len(axes) > image.ndim or len(axes) > len(canonical):
            raise ValueError(
                f"axes={axes!r} doesn't fit image of rank {image.ndim}"
            )
        for ch in axes:
            if ch not in canonical:
                raise ValueError(
                    f"axes={axes!r}: unknown axis '{ch}' "
                    f"(use chars from 'tczyx')"
                )
        ax_tuple = tuple(image.ndim - len(axes) + i for i in range(len(axes)))
    else:
        ax_tuple = tuple(int(a) for a in axes)

    out = [image]
    current = image
    while True:
        next_shape = list(current.shape)
        for ax in ax_tuple:
            next_shape[ax] = max(1, current.shape[ax] // 2)
        # Stop conditions.
        if levels is not None and len(out) >= levels:
            break
        if levels is None:
            # Auto-stop when any downsampled axis would fall below min_size.
            if any(next_shape[ax] < min_size for ax in ax_tuple):
                break
            # Also stop if there's nothing left to halve.
            if all(current.shape[ax] <= 1 for ax in ax_tuple):
                break
        if all(current.shape[ax] <= 1 for ax in ax_tuple):
            break
        current = _downsample_2x_meanpool(current, ax_tuple)
        out.append(current)
    return out


__all__ = ["make_pyramid_levels"]
