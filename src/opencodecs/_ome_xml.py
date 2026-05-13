"""OME-XML generator for OME-TIFF output.

The OME data model (https://www.openmicroscopy.org/Schemas/OME/) is
huge — experimenters, instruments, ROIs, masks, screens, plates,
detectors. For most scientific imaging the *useful* subset is small:

  * Image: SizeX/Y/Z/C/T, DimensionOrder, PixelType, PhysicalSizeX/Y/Z
  * Channel: Name, SamplesPerPixel, Color (ARGB), emission/excitation
    wavelengths
  * TiffData: per-plane → IFD mapping (only needed if planes don't
    follow DimensionOrder strictly)
  * Plane: TheC/TheZ/TheT, PositionX/Y/Z, DeltaT, ExposureTime

This module emits a valid OME-XML 2016-06 document covering that
subset. For the full schema users can still hand-author XML and pass
it via :class:`opencodecs.TiffWriter`'s ``metadata=`` kwarg, or
delegate to ``tifffile.OmeXml`` (a 1000-line implementation of the
full spec).

The output is well-formed for tifffile / Bio-Formats / QuPath /
ImageJ-OMERO readers; it is not minified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
from xml.sax.saxutils import escape as _xml_escape

import numpy as np


# OME XML namespace + schema URI for the 2016-06 release.
_OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
_OME_XSI = "http://www.w3.org/2001/XMLSchema-instance"
_OME_SCHEMA_LOC = (
    f"{_OME_NS} "
    "http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd"
)


_NP_TO_OME_PIXEL_TYPE = {
    "int8":    "int8",
    "int16":   "int16",
    "int32":   "int32",
    "uint8":   "uint8",
    "uint16":  "uint16",
    "uint32":  "uint32",
    "float32": "float",
    "float64": "double",
    "complex64":  "complex",
    "complex128": "double-complex",
}


def _pixel_type_for(dtype) -> str:
    """Map a numpy dtype to its OME PixelType string."""
    name = np.dtype(dtype).name
    try:
        return _NP_TO_OME_PIXEL_TYPE[name]
    except KeyError as e:
        raise ValueError(
            f"OME-XML: unsupported pixel type {name!r}; "
            f"supported: {sorted(_NP_TO_OME_PIXEL_TYPE)}"
        ) from e


@dataclass
class Channel:
    """One channel descriptor for OME-XML emission.

    All fields are optional; only ``samples_per_pixel`` defaults to 1.
    ``color`` accepts either an ARGB tuple ``(a, r, g, b)`` (each 0-255)
    or a signed 32-bit integer in OME's packed-int form (``(a<<24) |
    (r<<16) | (g<<8) | b`` interpreted as a Python int — negative when
    A is set high).
    """

    name: str | None = None
    samples_per_pixel: int = 1
    color: tuple[int, int, int, int] | int | None = None
    emission_wavelength_nm: float | None = None
    excitation_wavelength_nm: float | None = None


@dataclass
class Plane:
    """One ``<Plane>`` entry: maps a (C, Z, T) triple to acquisition
    metadata (stage position, dwell time, etc). Optional."""

    the_c: int
    the_z: int
    the_t: int
    position_x_um: float | None = None
    position_y_um: float | None = None
    position_z_um: float | None = None
    delta_t_seconds: float | None = None
    exposure_time_seconds: float | None = None


def _color_to_int(c) -> int:
    """Normalize a color tuple/int to OME's signed-int32 form."""
    if isinstance(c, int):
        return c
    if isinstance(c, (tuple, list)) and len(c) == 4:
        a, r, g, b = (int(v) & 0xFF for v in c)
        packed = (a << 24) | (r << 16) | (g << 8) | b
        # OME stores it as a signed 32-bit int (negative when alpha
        # has the high bit set).
        if packed & 0x80000000:
            packed -= 0x100000000
        return packed
    raise ValueError(f"OME-XML: bad color {c!r} (need (A,R,G,B) tuple or int32)")


def _fmt_float(v: float | None) -> str | None:
    """Format a float for OME-XML attribute value, dropping trailing
    zeros for readability but never using scientific notation."""
    if v is None:
        return None
    if v == int(v):
        return f"{int(v)}"
    return f"{v:.6f}".rstrip("0").rstrip(".")


def make_ome_xml(
    *,
    size_x: int,
    size_y: int,
    size_z: int = 1,
    size_c: int = 1,
    size_t: int = 1,
    dtype,
    dimension_order: str = "XYCZT",
    physical_size_x_um: float | None = None,
    physical_size_y_um: float | None = None,
    physical_size_z_um: float | None = None,
    channels: Sequence[Channel] | None = None,
    planes: Sequence[Plane] | None = None,
    image_name: str = "Image:0",
    image_id: str = "Image:0",
) -> str:
    """Build a single-Image OME-XML 2016-06 document.

    Parameters cover the practical scientific-imaging subset
    (dimension layout + per-channel metadata + per-plane acquisition
    data). For richer constructs (experimenters, instruments,
    detectors, ROIs) compose hand-written XML or use tifffile.OmeXml.

    ``dtype`` can be a numpy dtype, a numpy array, a numpy dtype name
    string, or any object with ``.dtype``.

    Returns
    -------
    str
        Well-formed UTF-8 XML, suitable for the OME-TIFF
        ImageDescription tag.
    """
    # Accept numpy arrays + dtypes + numpy scalar types + strings. Scalar
    # types (np.uint16, np.float32) have a class-level ``dtype`` attribute
    # that is not itself a dtype — handle that case explicitly.
    if isinstance(dtype, np.ndarray):
        dtype = dtype.dtype
    pixel_type = _pixel_type_for(dtype)

    valid_orders = {
        "XYCZT", "XYCTZ", "XYZCT", "XYZTC", "XYTCZ", "XYTZC",
    }
    if dimension_order not in valid_orders:
        raise ValueError(
            f"OME-XML: dimension_order must be one of {sorted(valid_orders)}; "
            f"got {dimension_order!r}"
        )
    if size_c < 1 or size_z < 1 or size_t < 1:
        raise ValueError("Size{C,Z,T} must be >= 1")
    if channels is not None and len(channels) != size_c:
        raise ValueError(
            f"channels has {len(channels)} entries but size_c={size_c}"
        )

    # Compute SamplesPerPixel total — if any channel uses >1 sample,
    # that goes in the Channel's @SamplesPerPixel.
    if channels is None:
        channels = [Channel() for _ in range(size_c)]

    pixels_attrs = {
        "ID": "Pixels:0",
        "DimensionOrder": dimension_order,
        "Type": pixel_type,
        "SizeX": str(size_x),
        "SizeY": str(size_y),
        "SizeZ": str(size_z),
        "SizeC": str(size_c),
        "SizeT": str(size_t),
        "Interleaved": "false",
        "BigEndian": "false",
    }
    if physical_size_x_um:
        pixels_attrs["PhysicalSizeX"] = _fmt_float(physical_size_x_um)
        pixels_attrs["PhysicalSizeXUnit"] = "µm"
    if physical_size_y_um:
        pixels_attrs["PhysicalSizeY"] = _fmt_float(physical_size_y_um)
        pixels_attrs["PhysicalSizeYUnit"] = "µm"
    if physical_size_z_um:
        pixels_attrs["PhysicalSizeZ"] = _fmt_float(physical_size_z_um)
        pixels_attrs["PhysicalSizeZUnit"] = "µm"

    out: list[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append(
        f'<OME xmlns="{_OME_NS}" xmlns:xsi="{_OME_XSI}" '
        f'xsi:schemaLocation="{_OME_SCHEMA_LOC}">'
    )
    out.append(
        f'  <Image ID="{_xml_escape(image_id)}" '
        f'Name="{_xml_escape(image_name)}">'
    )
    out.append("    <Pixels " + " ".join(
        f'{k}="{v}"' for k, v in pixels_attrs.items()
    ) + ">")

    # Channels.
    for i, ch in enumerate(channels):
        ch_attrs = [f'ID="Channel:0:{i}"',
                    f'SamplesPerPixel="{ch.samples_per_pixel}"']
        if ch.name is not None:
            ch_attrs.append(f'Name="{_xml_escape(ch.name)}"')
        if ch.color is not None:
            ch_attrs.append(f'Color="{_color_to_int(ch.color)}"')
        if ch.emission_wavelength_nm is not None:
            ch_attrs.append(
                f'EmissionWavelength="{_fmt_float(ch.emission_wavelength_nm)}"'
            )
            ch_attrs.append('EmissionWavelengthUnit="nm"')
        if ch.excitation_wavelength_nm is not None:
            ch_attrs.append(
                f'ExcitationWavelength="{_fmt_float(ch.excitation_wavelength_nm)}"'
            )
            ch_attrs.append('ExcitationWavelengthUnit="nm"')
        out.append("      <Channel " + " ".join(ch_attrs) + " />")

    # TiffData: one element with the default 1:1 plane:IFD mapping.
    # Bio-Formats parses this; without it readers fall back to the
    # DimensionOrder + Size* hints which already round-trip for
    # straightforward acquisitions.
    out.append('      <TiffData />')

    # Planes (optional).
    if planes:
        for pl in planes:
            attrs = [f'TheC="{pl.the_c}"',
                     f'TheZ="{pl.the_z}"',
                     f'TheT="{pl.the_t}"']
            if pl.position_x_um is not None:
                attrs.append(
                    f'PositionX="{_fmt_float(pl.position_x_um)}"'
                )
                attrs.append('PositionXUnit="µm"')
            if pl.position_y_um is not None:
                attrs.append(
                    f'PositionY="{_fmt_float(pl.position_y_um)}"'
                )
                attrs.append('PositionYUnit="µm"')
            if pl.position_z_um is not None:
                attrs.append(
                    f'PositionZ="{_fmt_float(pl.position_z_um)}"'
                )
                attrs.append('PositionZUnit="µm"')
            if pl.delta_t_seconds is not None:
                attrs.append(
                    f'DeltaT="{_fmt_float(pl.delta_t_seconds)}"'
                )
                attrs.append('DeltaTUnit="s"')
            if pl.exposure_time_seconds is not None:
                attrs.append(
                    f'ExposureTime="{_fmt_float(pl.exposure_time_seconds)}"'
                )
                attrs.append('ExposureTimeUnit="s"')
            out.append("      <Plane " + " ".join(attrs) + " />")

    out.append("    </Pixels>")
    out.append("  </Image>")
    out.append("</OME>")
    return "\n".join(out)


def write_ome_tiff(
    dest,
    image: np.ndarray,
    *,
    axes: str | None = None,
    channels: Sequence[Channel] | None = None,
    planes: Sequence[Plane] | None = None,
    physical_size_um: tuple[float, ...] | None = None,
    tile: tuple[int, int] | None = (256, 256),
    compression: str | int = "zstd",
    compression_level: int | None = None,
    bigtiff: bool = False,
    image_name: str = "Image:0",
) -> None:
    """Write a 2D/3D/4D/5D ndarray as a valid OME-TIFF.

    ``axes`` declares the dimension order of ``image``. Common forms:

      - 2D: ``YX``
      - 3D: ``ZYX``, ``CYX``, ``TYX``
      - 4D: ``CZYX``, ``ZCYX``, ``TCYX``, ``CTYX``
      - 5D: ``TCZYX``, ``CTZYX``, etc.

    If omitted, axes are inferred from ``image.ndim``:

      - 2 → ``YX``
      - 3 → ``ZYX``
      - 4 → ``CZYX``
      - 5 → ``TCZYX``

    Pages are emitted in the order that makes the OME ``DimensionOrder``
    self-consistent. Reading back via ``tifffile.imread`` or
    Bio-Formats yields the same ndarray with the same axes label.

    ``physical_size_um`` may be a tuple of length 2 (X, Y), 3 (X, Y, Z)
    or matching the spatial-axes count.
    """
    if image.ndim < 2 or image.ndim > 5:
        raise ValueError(
            f"write_ome_tiff: image must be 2-5D; got ndim={image.ndim}"
        )
    if axes is None:
        axes = {2: "YX", 3: "ZYX", 4: "CZYX", 5: "TCZYX"}[image.ndim]
    if len(axes) != image.ndim:
        raise ValueError(
            f"axes={axes!r} length doesn't match image.ndim={image.ndim}"
        )
    axes = axes.upper()
    if axes[-2:] != "YX":
        raise ValueError(
            f"axes must end in 'YX' (height, width); got {axes!r}"
        )
    valid_axes = set("XYZCT")
    if not all(a in valid_axes for a in axes):
        raise ValueError(
            f"axes characters must come from XYZCT; got {axes!r}"
        )
    if len(set(axes)) != len(axes):
        raise ValueError(f"axes has duplicate characters: {axes!r}")

    # Map ndim → SizeX/Y/Z/C/T.
    sizes = {a: 1 for a in "XYZCT"}
    for i, a in enumerate(axes):
        sizes[a] = int(image.shape[i])

    # OME DimensionOrder is always XY...; pick a remaining permutation
    # of CZT consistent with how planes are laid out in ``image``.
    # We emit pages in the order the array iterates over (everything
    # before YX), so DimensionOrder = "XY" + reverse(axes[:-2]).
    plane_axes = axes[:-2]  # e.g. "TCZ" or "CZ" etc.
    # The IFD ordering tifffile/Bio-Formats reads is fastest dim first
    # AFTER XY — and our loop yields planes by iterating in array
    # axis order (slowest first). So DimensionOrder = "XY" + reversed.
    dim_order = "XY" + "".join(reversed(plane_axes))
    # Pad to 5 chars by adding any missing dims at the end.
    for a in "CZT":
        if a not in dim_order:
            dim_order += a
    # Sanity: dim_order must be a permutation of XYZCT
    if sorted(dim_order) != list("CTXYZ"):
        raise ValueError(
            f"computed DimensionOrder {dim_order!r} isn't a permutation "
            f"of XYZCT (axes={axes!r})"
        )

    psx = psy = psz = None
    if physical_size_um is not None:
        if len(physical_size_um) == 2:
            psx, psy = physical_size_um
        elif len(physical_size_um) == 3:
            psx, psy, psz = physical_size_um
        else:
            raise ValueError(
                f"physical_size_um must have 2 or 3 entries; "
                f"got {len(physical_size_um)}"
            )

    xml = make_ome_xml(
        size_x=sizes["X"], size_y=sizes["Y"],
        size_z=sizes["Z"], size_c=sizes["C"], size_t=sizes["T"],
        dtype=image.dtype,
        dimension_order=dim_order,
        physical_size_x_um=psx,
        physical_size_y_um=psy,
        physical_size_z_um=psz,
        channels=channels,
        planes=planes,
        image_name=image_name,
    )

    # Flatten planes in array order (slowest-axis first) into a list
    # of 2-D images. The reshape preserves contiguity for the YX slabs.
    n_planes = int(np.prod(image.shape[:-2])) if image.ndim > 2 else 1
    flat = image.reshape((n_planes,) + tuple(image.shape[-2:]))

    from ._tiff_writer import TiffWriter
    with TiffWriter(dest, bigtiff=bigtiff) as w:
        for i in range(n_planes):
            w.write_page(
                flat[i], tile=tile,
                compression=compression,
                compression_level=compression_level,
                metadata=xml if i == 0 else None,
            )


__all__ = ["Channel", "Plane", "make_ome_xml", "write_ome_tiff"]
