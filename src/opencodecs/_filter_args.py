"""Argument checks shared by the filter codecs (predictors, shuffles, packing).

A filter's ``encode`` and ``decode`` used to end in ``**opts`` and never
look at it, so a keyword the codec does not implement (imagecodecs'
``runlen=`` or ``bitorder=`` for packints, ``axis=`` for byteshuffle, a
typo) was accepted and the call returned a different stream with no
error. The filters now name what they accept and raise on anything else.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def reject_unknown(label: str, opts: dict, hint: str = "") -> None:
    """Raise ``TypeError`` naming every keyword in ``opts``."""
    if opts:
        names = ", ".join(sorted(opts))
        raise TypeError(
            f"{label}: unexpected keyword argument(s) {names}"
            + (f"; {hint}" if hint else ""))


def as_input_array(data: Any) -> np.ndarray:
    """An ndarray as given, a bytes-like object as uint8, else ``np.asarray``.

    Bytes mean uint8, as they do in imagecodecs and in these codecs'
    decoders; ``np.asarray(b"...")`` would make a one-element ``S`` array.
    """
    if isinstance(data, np.ndarray):
        return data
    try:
        memoryview(data)
    except TypeError:
        return np.asarray(data)
    return np.frombuffer(data, dtype=np.uint8)


__all__ = ["reject_unknown", "as_input_array"]
