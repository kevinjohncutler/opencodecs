"""Optional byte-bounded cache of decoded tiles, with coordinated misses.

A source-byte cache avoids fetching bytes again but not decompressing them.
This one keeps decoded tiles for repeated or nearby regions, spending memory
to save computation. It is opt-in and off by default; there is no workload
evidence that it should be on for everyone.

Contract, in order of what goes wrong when it is broken:

* Keys must identify the source instance, the tile, and every decode option
  that changes the pixels. A key that omits the source would serve one
  file's tile for another.
* Cached arrays are read-only and are handed out directly. Callers that
  need to mutate must copy. Nothing here returns the storage it owns as
  writable.
* Source mutation during a session is not supported. A caller who changes
  the file must call ``clear()``; nothing detects it.
* Concurrent misses for one key are coordinated: the first claimant decodes,
  the others wait for it, and a failed claimant releases the key so waiters
  decode for themselves. Claim keys in a consistent (sorted) order across
  threads, or two threads holding claims can wait on each other.
* Retained bytes are counted here, separately from source caches and from
  pending pipeline reservations. An item larger than the whole budget is
  never stored.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np


class DecodedTileCache:
    """Least-recently-used cache of decoded tiles under a byte budget."""

    def __init__(self, max_bytes: int):
        max_bytes = int(max_bytes)
        if max_bytes <= 0:
            raise ValueError("decoded tile cache needs a positive byte budget")
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._entries: OrderedDict = OrderedDict()   # key -> (array, nbytes)
        self._inflight: dict = {}                    # key -> Event
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.stores = 0
        self.skipped_oversized = 0
        self.waits = 0

    # ------------------------------------------------------------- lookup
    def claim(self, key):
        """Return ``("hit", array)`` or ``("miss", None)``.

        A miss registers the caller as the decoder for ``key``; it must then
        ``store`` or ``release``. If another thread already holds the claim,
        this waits for it and retries, so one tile is never decoded twice at
        once.
        """
        while True:
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None:
                    self._entries.move_to_end(key)
                    self.hits += 1
                    return "hit", entry[0]
                event = self._inflight.get(key)
                if event is None:
                    self._inflight[key] = threading.Event()
                    self.misses += 1
                    return "miss", None
                self.waits += 1
            event.wait()

    def store(self, key, array: np.ndarray) -> np.ndarray:
        """Insert a decoded tile for a claimed key and release its waiters.

        The array is made read-only before it is shared. Returns the array
        that is now cached (the same object).
        """
        array.flags.writeable = False
        nbytes = int(array.nbytes)
        with self._lock:
            event = self._inflight.pop(key, None)
            if nbytes <= self.max_bytes:
                old = self._entries.pop(key, None)
                if old is not None:
                    self.bytes -= old[1]
                self._entries[key] = (array, nbytes)
                self.bytes += nbytes
                self.stores += 1
                while self.bytes > self.max_bytes:
                    _, (_, size) = self._entries.popitem(last=False)
                    self.bytes -= size
                    self.evictions += 1
            else:
                self.skipped_oversized += 1
        if event is not None:
            event.set()
        return array

    def release(self, key) -> None:
        """Give up a claim without storing, so waiters decode for themselves."""
        with self._lock:
            event = self._inflight.pop(key, None)
        if event is not None:
            event.set()

    # ------------------------------------------------------------ control
    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.bytes = 0

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        return {"entries": len(self._entries), "bytes": self.bytes,
                "max_bytes": self.max_bytes, "hits": self.hits,
                "misses": self.misses, "waits": self.waits,
                "evictions": self.evictions, "stores": self.stores,
                "skipped_oversized": self.skipped_oversized}


__all__ = ["DecodedTileCache"]
