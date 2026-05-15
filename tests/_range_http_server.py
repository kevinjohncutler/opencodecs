"""Tiny range-supporting HTTP server for tests.

Python's stdlib SimpleHTTPRequestHandler does NOT honor Range
requests — it always returns the whole file with a 200 status.
That makes it impossible to verify our native readers actually
do partial reads. This module spins up a server that DOES honor
``Range: bytes=A-B`` (and ``bytes=-N`` for tail reads), returning
a proper 206 with Content-Range. It also tracks how many bytes
the SERVER actually transmitted for each request so tests can
assert byte-savings claims.

Usage::

    from _range_http_server import range_http_server
    with range_http_server(directory) as (url_base, tracker):
        # ... do reads against url_base/file.nd2 ...
        assert tracker.bytes_served < tracker.file_size * 0.1
"""

from __future__ import annotations

import http.server
import os
import re
import socketserver
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class _RangeTracker:
    """Per-server counters. Read these from the test to verify how
    little the server actually had to serve."""
    bytes_served: int = 0
    requests: int = 0
    range_requests: int = 0
    full_requests: int = 0
    file_size: int = 0
    per_request: list[tuple[str, int]] = field(default_factory=list)


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    tracker: _RangeTracker = None  # type: ignore[assignment]
    directory: str = ""

    def log_message(self, *_):
        # Silence the default per-request stderr logging
        pass

    def do_HEAD(self):
        path = self._resolved_path()
        if not path or not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        path = self._resolved_path()
        if not path or not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        self.tracker.requests += 1
        self.tracker.file_size = size

        range_hdr = self.headers.get("Range")
        if range_hdr:
            start, end = self._parse_range(range_hdr, size)
            if start is None:
                self.send_error(416)   # Range Not Satisfiable
                return
            length = end - start + 1
            with open(path, "rb") as f:
                f.seek(start)
                body = f.read(length)
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header(
                "Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(body)
            self.tracker.bytes_served += len(body)
            self.tracker.range_requests += 1
            self.tracker.per_request.append((range_hdr, len(body)))
        else:
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(body)
            self.tracker.bytes_served += len(body)
            self.tracker.full_requests += 1
            self.tracker.per_request.append(("(full)", len(body)))

    @staticmethod
    def _parse_range(hdr: str, size: int) -> tuple[int | None, int | None]:
        """Parse a single ``bytes=A-B`` or ``bytes=-N`` Range header.
        Returns (start, end) inclusive, or (None, None) on parse fail."""
        m = re.match(r"bytes=(\d*)-(\d*)$", hdr.strip())
        if not m:
            return None, None
        a, b = m.group(1), m.group(2)
        if a == "" and b == "":
            return None, None
        if a == "":
            # Suffix range: last N bytes
            n = int(b)
            if n <= 0 or n > size:
                n = min(n, size)
            return size - n, size - 1
        start = int(a)
        end = int(b) if b else size - 1
        if start >= size or end >= size or start > end:
            return None, None
        return start, end

    def _resolved_path(self) -> str | None:
        # /file.tif → directory/file.tif
        rel = self.path.lstrip("/")
        full = os.path.normpath(os.path.join(self.directory, rel))
        if not full.startswith(self.directory):
            return None
        return full


@contextmanager
def range_http_server(
    directory: str | Path,
) -> Iterator[tuple[str, _RangeTracker]]:
    """Spin up an HTTP server with proper Range support.

    Yields ``(base_url, tracker)``. The tracker accumulates server-
    side byte counters so tests can assert how little of the file
    actually crossed the wire. Server shuts down on context exit.
    """
    tracker = _RangeTracker()

    class _Handler(_RangeHandler):
        pass

    _Handler.tracker = tracker
    _Handler.directory = str(directory)

    httpd = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    port = httpd.server_address[1]
    thr = threading.Thread(target=httpd.serve_forever, daemon=True)
    thr.start()
    try:
        yield f"http://127.0.0.1:{port}", tracker
    finally:
        httpd.shutdown()
