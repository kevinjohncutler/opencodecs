"""Every vendored file has a recorded origin, and nobody edited one quietly.

This is the offline half of ci/check_vendor_drift.py. It needs no
network, so it runs everywhere, every push.

The failure it exists to prevent is not exotic. Vendored source drifts
silently: PLIO shipped a decoder that died with SIGBUS on malformed
input long after upstream cfitsio had added the bounds checks that
prevent it, and nothing in the repository was positioned to notice.
Recording a hash and an origin per file is what makes the online check
(`python ci/check_vendor_drift.py check`) able to say anything.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_vendored_files_are_declared_and_unmodified():
    """Runs the checker's offline `verify` and requires a clean result.

    A failure means one of:
      * a vendored file was edited without updating its recorded hash
        (run `python ci/check_vendor_drift.py freeze` if intended), or
      * a new vendored file has no entry in 3rdparty/VENDOR.toml, so
        nothing records whether it is ours, current, or years stale.
    """
    proc = subprocess.run(
        [sys.executable, str(ROOT / "ci" / "check_vendor_drift.py"), "verify"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, (
        "vendored source is undeclared or has changed:\n"
        + proc.stdout + proc.stderr
    )
