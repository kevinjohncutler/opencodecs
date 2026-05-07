"""Format-agnostic infrastructure (color, errors, streaming protocol)."""

from .color import ColorSpec, parse_color
from .errors import OpenCodecsError, JxlError

__all__ = ["ColorSpec", "parse_color", "OpenCodecsError", "JxlError"]
