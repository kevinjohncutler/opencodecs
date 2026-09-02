"""Fail the sdist build if any build-input file is missing from the tarball.

The 0.1.13 sdist shipped with zero .pyx files: setuptools collects
ext_modules sources only *after* setup.py has run cythonize() (so it
sees generated _foo.c, never the .pyx), and extensions disabled on the
sdist-build host contribute nothing at all. MANIFEST.in now lists every
build input explicitly; this script keeps that list honest by checking
the built tarball against git.

Usage: python ci/check_sdist_contents.py dist/*.tar.gz
"""

import subprocess
import sys
import tarfile

# Everything git tracks under these patterns must be in the sdist.
# Generated Cython output (src/opencodecs/codecs/_*.c / _*.cpp) is
# intentionally excluded from the sdist, but git doesn't track those
# files, so no carve-out is needed here.
REQUIRED_PATTERNS = [
    "src/opencodecs/**/*.py",
    "src/opencodecs/**/*.pyx",
    "src/opencodecs/**/*.pxd",
    "src/opencodecs/**/*.c",
    "src/opencodecs/**/*.cpp",
    "src/opencodecs/**/*.h",
    "src/opencodecs/**/*.hpp",
    "3rdparty/**/*.c",
    "3rdparty/**/*.h",
    "3rdparty/**/*.hpp",
    "patches/*.diff",
    "setup.py",
    "pyproject.toml",
    "bench/build_libjxl.sh",
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    sdist_path = sys.argv[1]

    tracked = set(
        subprocess.run(
            ["git", "ls-files", "--", *REQUIRED_PATTERNS],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )

    with tarfile.open(sdist_path) as tar:
        # Members are prefixed with the "opencodecs-X.Y.Z/" root dir.
        shipped = {
            name.split("/", 1)[1]
            for name in tar.getnames()
            if "/" in name
        }

    missing = sorted(tracked - shipped)
    if missing:
        print(f"{sdist_path} is missing {len(missing)} build input(s):")
        for path in missing:
            print(f"  {path}")
        return 1

    n_pyx = sum(1 for p in shipped if p.endswith(".pyx"))
    print(f"{sdist_path}: all {len(tracked)} build inputs present ({n_pyx} .pyx)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
