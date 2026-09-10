"""The version is written in two files and nothing kept them equal.

pyproject.toml is what the wheel is built with; opencodecs.__version__
is what a user sees at runtime. They are separate literals, so they can
drift, and the drift is invisible until someone reports a version that
does not match what they installed.

A tag push publishes straight to PyPI here, so a stale literal ships.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import opencodecs as oc

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def _pyproject_version() -> str:
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = pytest.importorskip("tomli")
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def test_runtime_version_matches_pyproject():
    assert oc.__version__ == _pyproject_version(), (
        f"opencodecs.__version__ is {oc.__version__} but pyproject.toml "
        f"says {_pyproject_version()}; a wheel built from this tree would "
        f"report the wrong version at runtime")


def test_version_looks_like_a_release():
    """Catches a placeholder or a half-finished edit."""
    v = _pyproject_version()
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[.-]?(?:a|b|rc|dev|post)\d*)?", v), (
        f"{v!r} is not a version this project's tags (v<x.y.z>) can carry")
