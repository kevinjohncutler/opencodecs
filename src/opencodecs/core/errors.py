"""Exception hierarchy for opencodecs."""

from __future__ import annotations


class OpenCodecsError(RuntimeError):
    """Base class for all opencodecs errors."""


class JxlError(OpenCodecsError):
    """Raised when libjxl signals an error."""


class JxlNeedMoreInput(OpenCodecsError):
    """Raised when the decoder needs more bytes than were provided."""
