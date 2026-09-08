"""Pyramid over a single JPEG 2000 codestream.

JPEG 2000's wavelet decomposition already contains every level: asking
openjpeg for ``cp_reduce=k`` makes it skip the k finest resolutions,
which it does by not entropy-decoding those subbands at all. So the
levels here are not built, they are read -- and reading a 256x256
overview out of a 2048x2048 codestream costs about 2 ms against 211 ms
for the full decode.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._scaled_pyramid import ScaledCodestreamPyramid, probe_by_reduction

# JPEG 2000 codestreams rarely carry more than 6 decomposition levels
# and openjpeg refuses beyond what is there, so this only bounds the
# probe loop; the real limit comes from the codestream.
_MAX_REDUCTION = 12


class Jpeg2kPyramidReader(ScaledCodestreamPyramid):
    """Multi-resolution view of one JP2 / J2K codestream."""

    codec_name = "jpeg2k"

    def __init__(self, data: Any, *, numthreads: int | None = None,
                 max_levels: int | None = None):
        super().__init__(data, max_levels=max_levels)
        self._numthreads = numthreads

    def _probe_levels(self):
        from .codecs import _jpeg2k

        def info(r):
            d = _jpeg2k.decode_info(self._data, reduce=r)
            return d["shape"], np.dtype(d["dtype"])

        levels = probe_by_reduction(info, max_reduction=_MAX_REDUCTION)
        if self._max_levels is not None:
            levels = levels[: self._max_levels]
        return levels

    def _decode_level(self, reduction: int) -> np.ndarray:
        from .codecs import _jpeg2k
        return _jpeg2k.decode(self._data, reduce=reduction,
                              numthreads=self._numthreads)
