"""Profile a real tifffile decode to find the actual bottleneck."""

from __future__ import annotations

import cProfile
import os
import pstats
import tempfile

import numpy as np
import tifffile


def main() -> None:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 65536, (8192, 8192), dtype=np.uint16)

    fd, path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    tifffile.imwrite(path, arr, compression="zstd", tile=(256, 256))
    size = os.path.getsize(path)
    print(f"file: {size/1e6:.1f} MB, tiled 256x256")

    # warmup
    _ = tifffile.imread(path)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        _ = tifffile.imread(path)
    pr.disable()

    print()
    stats = pstats.Stats(pr).sort_stats("cumulative")
    stats.print_stats(25)

    print()
    print("=== by total time ===")
    stats.sort_stats("tottime").print_stats(25)

    os.unlink(path)


if __name__ == "__main__":
    main()
