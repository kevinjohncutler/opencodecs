"""Pyramid over a single HTJ2K codestream.

Same idea as the JPEG 2000 reader: OpenJPH's
``restrict_input_resolution`` skips the finest resolutions before
``create()``, so the subbands are never read. The codestream reports
its own ceiling as ``num_decompositions``, which is the largest useful
reduction and saves probing past the end.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._scaled_pyramid import ScaledCodestreamPyramid, probe_by_reduction


class Htj2kPyramidReader(ScaledCodestreamPyramid):
    """Multi-resolution view of one HTJ2K codestream."""

    codec_name = "htj2k"

    def __init__(self, data: Any, *, ignore_unsupported: bool = False,
                 max_levels: int | None = None):
        super().__init__(data, max_levels=max_levels)
        self._ignore_unsupported = ignore_unsupported

    def _probe_levels(self):
        from .codecs import _openjph

        head = _openjph.decode_info(self._data)
        ceiling = int(head.get("num_decompositions", 0))

        def info(r):
            d = _openjph.decode_info(self._data, reduce=r)
            comps = d["components"]
            shape = ((d["height"], d["width"]) if comps == 1
                     else (d["height"], d["width"], comps))
            # The decoder's own choice (it covers 32-bit and NLT float
            # codestreams), so a level's advertised dtype matches what
            # reading it returns.
            if d["dtype"] is None:
                raise ValueError(
                    "HTJ2K: the components differ in precision, sign, "
                    "nonlinearity or subsampling; not supported")
            return shape, np.dtype(d["dtype"])

        levels = probe_by_reduction(info, max_reduction=ceiling)
        if self._max_levels is not None:
            levels = levels[: self._max_levels]
        return levels

    def _decode_level(self, reduction: int) -> np.ndarray:
        from .codecs import _openjph
        # planar=False: the levels advertise (H, W, C), whether or not
        # the codestream uses the component transform.
        return _openjph.decode(
            self._data, reduce=reduction, planar=False,
            ignore_unsupported=self._ignore_unsupported)
