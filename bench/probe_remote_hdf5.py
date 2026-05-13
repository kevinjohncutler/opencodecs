"""Smoke-test remote HDF5 reads against a user-supplied URL.

Usage::

    python bench/probe_remote_hdf5.py <url> [<dataset-path>]

Prints:
  * total bytes transferred (no prefetch vs with prefetch)
  * wall time for both
  * number of HTTP requests for both
  * confirmation that h5py read the data correctly

The local fixture test in `tests/test_hdf5_http.py` covers the
plumbing (Range-aware test server in a thread). This script
exists for ad-hoc verification against an actual cloud HDF5 file
on the user's hand — public NASA / SnowEx / IDC URLs tend to drift,
so we don't bake any into the test suite.

Example URLs that have worked in the past (verify before using):
  - https://its-live-data.s3.amazonaws.com/<some-velocity-mosaic>.h5
  - https://nsidc-cumulus-prod-protected.s3.amazonaws.com/ATLAS/<...>.h5

Tip: any S3 object with HTTP/1.1 Range support works. Most modern
public-data buckets do.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    url = argv[1]
    dataset_path = argv[2] if len(argv) >= 3 else None

    # Allow running from source tree without `pip install -e .`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from opencodecs._hdf5_http import (
        _SOURCE_REGISTRY, open_remote_hdf5, prefetch_hdf5_chunks,
    )

    print(f"URL: {url}")

    # Round 1: no prefetch.
    t0 = time.perf_counter()
    with open_remote_hdf5(url) as f:
        src = _SOURCE_REGISTRY[f.id]
        ds_name = dataset_path or _first_dataset(f)
        if ds_name is None:
            print("  no datasets found in the file", file=sys.stderr)
            return 1
        print(f"dataset: {ds_name}")
        d = f[ds_name]
        print(f"  shape={d.shape}  dtype={d.dtype}  chunks={d.chunks}")
        # Read a small slice — first chunk.
        sl = _first_chunk_slice(d)
        print(f"  slice: {sl}")
        arr_no_prefetch = d[sl]
        elapsed_no = time.perf_counter() - t0
        stats_no = dict(src.stats)
    print(
        f"  no prefetch: {elapsed_no*1000:.1f} ms, "
        f"{stats_no['requests']} reqs, "
        f"{stats_no['bytes_fetched']:,} bytes"
    )

    # Round 2: with prefetch.
    t0 = time.perf_counter()
    with open_remote_hdf5(url, max_workers=8) as f:
        src = _SOURCE_REGISTRY[f.id]
        d = f[dataset_path or ds_name]
        sl = _first_chunk_slice(d)
        n_chunks = prefetch_hdf5_chunks(d, sl)
        arr_prefetch = d[sl]
        elapsed_yes = time.perf_counter() - t0
        stats_yes = dict(src.stats)
    print(
        f"  with prefetch ({n_chunks} chunks): "
        f"{elapsed_yes*1000:.1f} ms, "
        f"{stats_yes['requests']} reqs, "
        f"{stats_yes['bytes_fetched']:,} bytes"
    )

    # Same values both ways?
    import numpy as np
    assert np.array_equal(arr_no_prefetch, arr_prefetch), \
        "prefetch path produced different bytes — bug"
    print(f"  values bit-exact match between paths: ok")
    print()
    print(
        f"prefetch saved {stats_no['requests'] - stats_yes['requests']} "
        f"HTTP round trips "
        f"({stats_no['requests']/stats_yes['requests']:.1f}x fewer)"
    )
    return 0


def _first_dataset(grp) -> str | None:
    """Return the path of the first dataset in the file (depth-first)."""
    import h5py
    for key in grp:
        obj = grp[key]
        if isinstance(obj, h5py.Dataset):
            return obj.name
        if isinstance(obj, h5py.Group):
            sub = _first_dataset(obj)
            if sub is not None:
                return sub
    return None


def _first_chunk_slice(ds):
    """Build a slice covering the first chunk of ``ds``."""
    if ds.chunks is None:
        # Contiguous dataset — read a small leading slab so we don't
        # pull the whole thing.
        head = tuple(min(64, s) for s in ds.shape)
        return tuple(slice(0, n) for n in head)
    return tuple(slice(0, c) for c in ds.chunks)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
