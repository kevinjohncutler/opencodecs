"""Unified color encoding spec.

The same `ColorSpec` is consumed by every codec; each codec maps it to its
native enum. v0.1 only wires JPEG XL.

Names follow CSS Color 4 / ITU-R conventions:
    "srgb"           sRGB primaries + sRGB transfer
    "srgb-linear"    sRGB primaries + linear transfer
    "display-p3"     P3 primaries + sRGB transfer (Apple's Display P3)
    "p3-linear"      P3 primaries + linear transfer
    "rec2020-pq"     BT.2020 primaries + SMPTE 2084 (PQ) transfer
    "rec2020-hlg"    BT.2020 primaries + Hybrid Log-Gamma transfer
    "rec2020-linear" BT.2020 primaries + linear transfer
    "rec709"         BT.709 primaries (== sRGB) + BT.709 transfer

Aliases: "p3" -> "display-p3", "bt2020-*" -> "rec2020-*", "linear" -> "srgb-linear".

The integer enum values match libjxl's JxlPrimaries / JxlTransferFunction so
this struct can be passed straight through to the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass


# JxlPrimaries
JXL_PRIMARIES_SRGB = 1
JXL_PRIMARIES_CUSTOM = 2
JXL_PRIMARIES_2100 = 9
JXL_PRIMARIES_P3 = 11

# JxlTransferFunction
JXL_TF_709 = 1
JXL_TF_UNKNOWN = 2
JXL_TF_LINEAR = 8
JXL_TF_SRGB = 13
JXL_TF_PQ = 16
JXL_TF_DCI = 17
JXL_TF_HLG = 18
JXL_TF_GAMMA = 65535  # custom; gamma value carried separately

# JxlWhitePoint
JXL_WP_D65 = 1
JXL_WP_CUSTOM = 2
JXL_WP_E = 10
JXL_WP_DCI = 11

# JxlRenderingIntent
JXL_RI_PERCEPTUAL = 0
JXL_RI_RELATIVE = 1
JXL_RI_SATURATION = 2
JXL_RI_ABSOLUTE = 3


@dataclass(frozen=True)
class ColorSpec:
    """A complete color encoding spec, codec-agnostic.

    Attributes use libjxl enum integers so codecs that wrap libjxl can pass
    through directly. Other codecs translate as needed.
    """

    primaries: int = JXL_PRIMARIES_SRGB
    transfer: int = JXL_TF_SRGB
    white_point: int = JXL_WP_D65
    rendering_intent: int = JXL_RI_RELATIVE
    gamma: float = 0.0  # only used when transfer == JXL_TF_GAMMA

    @property
    def is_hdr(self) -> bool:
        return self.transfer in (JXL_TF_PQ, JXL_TF_HLG)


SRGB = ColorSpec()
SRGB_LINEAR = ColorSpec(transfer=JXL_TF_LINEAR)
DISPLAY_P3 = ColorSpec(primaries=JXL_PRIMARIES_P3, transfer=JXL_TF_SRGB)
P3_LINEAR = ColorSpec(primaries=JXL_PRIMARIES_P3, transfer=JXL_TF_LINEAR)
REC709 = ColorSpec(transfer=JXL_TF_709)
REC2020_PQ = ColorSpec(primaries=JXL_PRIMARIES_2100, transfer=JXL_TF_PQ)
REC2020_HLG = ColorSpec(primaries=JXL_PRIMARIES_2100, transfer=JXL_TF_HLG)
REC2020_LINEAR = ColorSpec(primaries=JXL_PRIMARIES_2100, transfer=JXL_TF_LINEAR)


_NAMED: dict[str, ColorSpec] = {
    "srgb": SRGB,
    "srgb-linear": SRGB_LINEAR,
    "linear-srgb": SRGB_LINEAR,
    "linear": SRGB_LINEAR,
    "display-p3": DISPLAY_P3,
    "p3": DISPLAY_P3,
    "p3-linear": P3_LINEAR,
    "rec709": REC709,
    "bt709": REC709,
    "rec2020-pq": REC2020_PQ,
    "bt2020-pq": REC2020_PQ,
    "rec2020-hlg": REC2020_HLG,
    "bt2020-hlg": REC2020_HLG,
    "rec2020-linear": REC2020_LINEAR,
    "bt2020-linear": REC2020_LINEAR,
}


def parse_color(spec: str | ColorSpec | None) -> ColorSpec | None:
    """Resolve a color spec string or ColorSpec to a ColorSpec.

    None is passed through (codec uses its default).
    """
    if spec is None or isinstance(spec, ColorSpec):
        return spec
    if isinstance(spec, str):
        key = spec.strip().lower().replace("_", "-")
        if key in _NAMED:
            return _NAMED[key]
        raise ValueError(
            f"unknown color spec {spec!r}; known: {sorted(_NAMED)}"
        )
    raise TypeError(f"color must be str or ColorSpec, not {type(spec).__name__}")
