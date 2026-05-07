"""Helper for codec modules with optional Cython backends.

Each codec in opencodecs is implemented as a Cython extension that links
against an external C library (libzstd, libavif, libheif, ...). The
build is conditional — extensions whose system header is missing are
skipped — so on a given platform some extensions may not exist at
runtime.

Codec adapter modules (``opencodecs/_<name>_codec.py``) and Python-side
surfaces (``opencodecs.jxl``, ``opencodecs.parallel``) all need to
import from a possibly-missing extension. This helper standardises the
"try, fall back to clear-error stubs" idiom so:

* ``import opencodecs`` never raises on a platform where one or more
  codecs can't build
* Calling a function backed by an unavailable extension raises
  ``ImportError`` with a message that points at INSTALL.md

Usage::

    from .core._optional_backend import import_or_stubs

    encode, decode, check_signature, _HAVE_BACKEND = import_or_stubs(
        "opencodecs.codecs._zstd", "encode", "decode", "check_signature",
    )
"""

from __future__ import annotations

import importlib
from typing import Any


def _stub_factory(modname: str, attr: str, exc: BaseException):
    """Return a function that raises a clear ImportError when called.

    Used as a placeholder for symbols that would have come from a
    Cython extension whose backing library wasn't available.
    """
    def _stub(*_a, **_kw):
        raise ImportError(
            f"opencodecs codec {modname.rsplit('.', 1)[-1].lstrip('_')!r} "
            f"is not available on this build (cannot import {attr} from "
            f"{modname}): {exc}. "
            "See INSTALL.md for the system library required."
        )
    _stub.__name__ = attr
    _stub.__qualname__ = f"<missing>.{attr}"
    return _stub


def import_or_stubs(modname: str, *attrs: str) -> tuple[Any, ...]:
    """Import ``attrs`` from ``modname``; return them plus a backend-flag.

    Returns ``(*attr_values, have_backend)``. On success ``have_backend``
    is ``True`` and each ``attr_value`` is the real symbol. On
    ``ImportError`` ``have_backend`` is ``False`` and each ``attr_value``
    is a stub callable that raises ``ImportError`` with a clear message.
    """
    try:
        mod = importlib.import_module(modname)
    except ImportError as exc:
        return (*[_stub_factory(modname, a, exc) for a in attrs], False)
    return (*[getattr(mod, a) for a in attrs], True)


__all__ = ["import_or_stubs"]
