"""A small local DICOMweb server for tests.

It serves a synthetic study the way a PACS does over DICOMweb (DICOM
PS3.18): QIDO-RS searches return the DICOM JSON model, and WADO-RS
frame requests return a multipart/related body whose part carries the
transfer syntax. It replaces a live public demo server, which made the
client tests depend on someone else's uptime.

Content negotiation follows PS3.18 section 8.7.3 for frames:

* ``type="application/octet-stream"`` with no transfer-syntax parameter
  asks for Explicit VR Little Endian, so a compressed frame is served
  uncompressed (this server keeps each frame's pixels, so it can).
* ``transfer-syntax=*`` asks for the frame as stored.
* A compressed media type (``image/jls`` and so on) asks for the stored
  frame when it matches that type, and is refused with 406 otherwise.

The frame payloads are produced by reference encoders (imagecodecs,
pydicom), never by opencodecs itself, so a matching round trip means
two implementations agree. HTJ2K is the exception: no reference encoder
is installed, so its frames come from opencodecs and are checked against
OpenJPEG's decoder when the study is built.
"""

from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import numpy as np

EXPLICIT_VR_LE = "1.2.840.10008.1.2.1"
JPEG_BASELINE = "1.2.840.10008.1.2.4.50"
JPEGLS_LOSSLESS = "1.2.840.10008.1.2.4.80"
JPEG2K_LOSSLESS = "1.2.840.10008.1.2.4.90"
HTJ2K_LOSSLESS = "1.2.840.10008.1.2.4.201"
RLE_LOSSLESS = "1.2.840.10008.1.2.5"

# Frame media types, PS3.18 Table 8.7.3-2.
MEDIA_TYPE = {
    EXPLICIT_VR_LE: "application/octet-stream",
    JPEG_BASELINE: "image/jpeg",
    JPEGLS_LOSSLESS: "image/jls",
    JPEG2K_LOSSLESS: "image/jp2",
    HTJ2K_LOSSLESS: "image/jphc",
    RLE_LOSSLESS: "image/dicom-rle",
}

SECONDARY_CAPTURE = "1.2.840.10008.5.1.4.1.1.7"
PREFIX = "/dicom-web"


@dataclass
class Instance:
    """One SOP instance: its pixels, and each frame as stored."""
    uid: str
    transfer_syntax: str
    pixels: np.ndarray            # (frames, rows, columns[, samples])
    stored: list[bytes]           # one encoded payload per frame
    expected: np.ndarray          # what a correct decoder returns
    pixel_representation: int = 0
    bits_stored: int | None = None

    @property
    def rows(self) -> int:
        return self.pixels.shape[1]

    @property
    def columns(self) -> int:
        return self.pixels.shape[2]

    @property
    def samples_per_pixel(self) -> int:
        return self.pixels.shape[3] if self.pixels.ndim == 4 else 1

    @property
    def bits_allocated(self) -> int:
        return self.pixels.dtype.itemsize * 8

    def raw_frame(self, index: int) -> bytes:
        """The frame as Explicit VR Little Endian pixel data."""
        return self.pixels[index].astype(
            self.pixels.dtype.newbyteorder("<"), copy=False).tobytes()


@dataclass
class Series:
    uid: str
    modality: str
    instances: list[Instance] = field(default_factory=list)


@dataclass
class Study:
    uid: str
    patient_id: str
    series: list[Series] = field(default_factory=list)

    def instance(self, uid: str) -> Instance:
        for s in self.series:
            for i in s.instances:
                if i.uid == uid:
                    return i
        raise KeyError(uid)


def _el(vr: str, *values) -> dict:
    return {"vr": vr, "Value": list(values)}


def _study_json(study: Study) -> dict:
    return {
        "0020000D": _el("UI", study.uid),
        "00100020": _el("LO", study.patient_id),
        "00080061": _el("CS", *sorted({s.modality for s in study.series})),
        "00201206": _el("IS", len(study.series)),
        "00201208": _el("IS", sum(len(s.instances) for s in study.series)),
    }


def _series_json(study: Study, series: Series) -> dict:
    return {
        "0020000D": _el("UI", study.uid),
        "0020000E": _el("UI", series.uid),
        "00080060": _el("CS", series.modality),
        "00201209": _el("IS", len(series.instances)),
    }


def _instance_json(study: Study, series: Series, number: int,
                   inst: Instance) -> dict:
    return {
        "00080016": _el("UI", SECONDARY_CAPTURE),
        "00080018": _el("UI", inst.uid),
        "00083002": _el("UI", inst.transfer_syntax),
        "0020000D": _el("UI", study.uid),
        "0020000E": _el("UI", series.uid),
        "00200013": _el("IS", number),
        "00280002": _el("US", inst.samples_per_pixel),
        "00280008": _el("IS", inst.pixels.shape[0]),
        "00280010": _el("US", inst.rows),
        "00280011": _el("US", inst.columns),
        "00280100": _el("US", inst.bits_allocated),
        "00280101": _el("US", inst.bits_stored or inst.bits_allocated),
        "00280103": _el("US", inst.pixel_representation),
    }


def _media_params(value: str) -> tuple[str, dict[str, str]]:
    """Split one media-range into its type and lower-cased parameters."""
    fields = [f.strip() for f in value.split(";")]
    params = {}
    for f in fields[1:]:
        key, _, val = f.partition("=")
        params[key.strip().lower()] = val.strip().strip('"')
    return fields[0].lower(), params


def _negotiate_frame(accept: str, inst: Instance) -> tuple[str, str] | None:
    """(part media type, transfer syntax) to send, or None for 406."""
    for media_range in accept.split(","):
        mtype, params = _media_params(media_range)
        if mtype not in ("multipart/related", "*/*"):
            continue
        want = params.get("type", "application/octet-stream").lower()
        ts = params.get("transfer-syntax")
        if want == "application/octet-stream":
            if ts == "*" or ts == inst.transfer_syntax:
                return want, inst.transfer_syntax
            if ts in (None, EXPLICIT_VR_LE):
                return want, EXPLICIT_VR_LE
            continue
        if want in ("*/*", MEDIA_TYPE[inst.transfer_syntax]):
            if ts in (None, "*", inst.transfer_syntax):
                return MEDIA_TYPE[inst.transfer_syntax], inst.transfer_syntax
    return None


def _wants_json(accept: str) -> bool:
    return any(_media_params(r)[0] in ("application/dicom+json",
                                        "application/json", "*/*")
               for r in accept.split(",")) or not accept.strip()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    study: Study
    requests: list

    def log_message(self, *args):
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send(status, message.encode(), "text/plain")

    def do_GET(self):  # noqa: N802 (http.server naming)
        self.requests.append({"path": self.path, "headers": dict(self.headers)})
        path = self.path.split("?", 1)[0]
        if not path.startswith(PREFIX):
            return self._error(404, "not a DICOMweb path")
        path = path[len(PREFIX):]
        accept = self.headers.get("Accept", "")
        study = self.study
        ids = r"([0-9.]+)"
        m = re.fullmatch(rf"/studies/{ids}/series/{ids}/instances/{ids}/frames/([0-9,]+)", path)
        if m:
            return self._frames(*m.groups(), accept)
        qido = None
        if path == "/studies":
            qido = [_study_json(study)]
        elif m := re.fullmatch(rf"/studies/{ids}/series", path):
            if m.group(1) == study.uid:
                qido = [_series_json(study, s) for s in study.series]
        elif m := re.fullmatch(rf"/studies/{ids}/series/{ids}/instances", path):
            for s in study.series:
                if m.group(1) == study.uid and s.uid == m.group(2):
                    qido = [_instance_json(study, s, n + 1, i)
                            for n, i in enumerate(s.instances)]
        if qido is None:
            return self._error(404, "no such resource")
        if not _wants_json(accept):
            return self._error(406, "QIDO-RS answers in application/dicom+json")
        self._send(200, json.dumps(qido).encode(), "application/dicom+json")

    def _frames(self, study_uid, series_uid, instance_uid, numbers, accept):
        study = self.study
        series = next((s for s in study.series if s.uid == series_uid), None)
        if study_uid != study.uid or series is None:
            return self._error(404, "no such series")
        inst = next((i for i in series.instances if i.uid == instance_uid), None)
        if inst is None:
            return self._error(404, "no such instance")
        frames = [int(n) for n in numbers.split(",") if n]
        if not frames or any(not 1 <= n <= len(inst.stored) for n in frames):
            return self._error(404, "no such frame")
        chosen = _negotiate_frame(accept, inst)
        if chosen is None:
            return self._error(406, f"cannot serve {accept!r}")
        media, ts = chosen
        boundary = "oc-dicomweb-boundary"
        body = b""
        for n in frames:
            payload = (inst.stored[n - 1] if ts == inst.transfer_syntax
                       else inst.raw_frame(n - 1))
            body += (f"--{boundary}\r\nContent-Type: {media}; "
                     f'transfer-syntax="{ts}"\r\n\r\n').encode() + payload + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        self._send(200, body, f'multipart/related; type="{media}"; boundary={boundary}')


@contextmanager
def dicomweb_server(study: Study) -> Iterator[tuple[str, list]]:
    """Serve ``study``; yields (base URL, list of recorded requests)."""
    requests: list = []
    handler = type("Handler", (_Handler,), {"study": study, "requests": requests})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}{PREFIX}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# The synthetic study
# ---------------------------------------------------------------------------

# UUID-derived UIDs (the 2.25 root, PS3.5 B.2) need no registered org root.
_UID = "2.25.{}"


def _rle(pixels: np.ndarray, pixel_representation: int) -> bytes:
    from pydicom.pixels.encoders import RLELosslessEncoder
    spp = pixels.shape[2] if pixels.ndim == 3 else 1
    return RLELosslessEncoder.encode(
        pixels, rows=pixels.shape[0], columns=pixels.shape[1],
        samples_per_pixel=spp, bits_allocated=pixels.itemsize * 8,
        bits_stored=pixels.itemsize * 8, pixel_representation=pixel_representation,
        photometric_interpretation="RGB" if spp == 3 else "MONOCHROME2",
        number_of_frames=1, planar_configuration=0)


def synthetic_study(seed: int = 20260926) -> Study:
    """One study, two series, one instance per transfer syntax.

    Shapes are deliberately not square and not powers of two, so a
    transposed or mis-strided decode cannot pass. Needs imagecodecs and
    pydicom; the HTJ2K instance also needs opencodecs' OpenJPH codec.
    """
    import imagecodecs as ic

    rng = np.random.default_rng(seed)
    uid = iter(range(1000, 2000))
    new_uid = lambda: _UID.format(seed * 1000 + next(uid))  # noqa: E731

    def mono(dtype, lo, hi, frames=1):
        return rng.integers(lo, hi, (frames, 37, 53)).astype(dtype)

    def inst(ts, pixels, encode, *, expected=None, pixrep=0, bits_stored=None):
        stored = [bytes(encode(f)) for f in pixels]
        return Instance(new_uid(), ts, pixels, stored,
                        pixels if expected is None else expected,
                        pixrep, bits_stored)

    u12 = mono("<u2", 0, 4096)
    i16 = mono("<i2", -2000, 2000)
    rgb = rng.integers(0, 256, (1, 29, 41, 3)).astype("u1")
    gray8 = (mono("u1", 0, 256) // 16 * 16).astype("u1")
    multi = mono("u1", 0, 256, frames=5)

    j2k = lambda f: ic.jpeg2k_encode(  # noqa: E731
        f, level=0, reversible=True, codecformat=ic.JPEG2K.CODEC.J2K)
    jpeg = [ic.jpeg8_encode(f, level=90) for f in gray8]

    ct = Series(new_uid(), "CT", [
        inst(EXPLICIT_VR_LE, u12, lambda f: f.tobytes(), bits_stored=12),
        inst(EXPLICIT_VR_LE, i16, lambda f: f.tobytes(), pixrep=1),
        inst(JPEGLS_LOSSLESS, u12, ic.jpegls_encode, bits_stored=12),
        inst(JPEG2K_LOSSLESS, u12, j2k, bits_stored=12),
        inst(RLE_LOSSLESS, u12, lambda f: _rle(f, 0), bits_stored=12),
        inst(RLE_LOSSLESS, i16, lambda f: _rle(f, 1), pixrep=1),
        inst(JPEGLS_LOSSLESS, multi, ic.jpegls_encode),
    ])
    other = Series(new_uid(), "OT", [
        inst(EXPLICIT_VR_LE, rgb, lambda f: f.tobytes()),
        inst(RLE_LOSSLESS, rgb, lambda f: _rle(f, 0)),
        # Lossy: a correct decoder returns what libjpeg decodes, not the input.
        Instance(new_uid(), JPEG_BASELINE, gray8, jpeg,
                 np.stack([ic.jpeg8_decode(p) for p in jpeg])),
    ])
    try:
        from opencodecs.codecs import _openjph
    except ImportError:
        pass
    else:
        stored = [bytes(_openjph.encode(f)) for f in u12]
        for f, p in zip(u12, stored):
            # OpenJPEG vouches for the codestream; see the module docstring.
            assert np.array_equal(ic.jpeg2k_decode(p), f), "HTJ2K fixture is wrong"
        ct.instances.append(Instance(new_uid(), HTJ2K_LOSSLESS, u12, stored, u12,
                                     0, 12))
    return Study(new_uid(), "SYNTHETIC-0001", [ct, other])
