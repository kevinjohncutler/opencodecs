#!/usr/bin/env python3
"""Catch workflow `run:` blocks that will not survive YAML scalar folding.

The hazard is specific and it is invisible to a YAML linter, because the
YAML is valid -- the bug is what the scalar folds *into*:

    run: python -c "
      import x
      print(x)"

That is a plain scalar, so YAML joins the lines with spaces before the
shell ever sees them. The continuation lines' indentation lands inside
the Python string and the interpreter raises IndentationError on line 1.
Written with a literal block scalar instead, the newlines are preserved
and the same code is fine:

    run: |
      python -c "
      import x
      print(x)
      "

This shipped once and failed all six CI jobs at the same step. The check
is: any `run:` whose value starts on the key's own line and continues
onto the next is a folded scalar, and folded scalars must be single
commands. Literal (`|`) and folded (`>`) block markers are exempt, and
their contents get a `bash -n` syntax check instead.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import yaml

WORKFLOWS = (pathlib.Path(__file__).resolve().parent.parent
             / ".github" / "workflows")

_RUN = re.compile(r"^(?P<indent>\s*)run:[ \t]*(?P<value>\S.*)$")


def folded_multiline_runs(path: pathlib.Path) -> list[str]:
    """`run:` values that start inline and spill onto following lines."""
    problems = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        m = _RUN.match(line)
        if not m or m.group("value")[0] in "|>":
            continue
        key_indent = len(m.group("indent"))
        # A continuation is the next non-blank line indented past the key.
        for follow in lines[i + 1:]:
            if not follow.strip():
                continue
            indent = len(follow) - len(follow.lstrip())
            if indent > key_indent:
                problems.append(
                    f"{path.name}:{i + 1}: `run:` starts on its own line and "
                    f"continues onto the next, so YAML folds them into one "
                    f"line. Use `run: |` instead.\n    {line.strip()[:90]}")
            break
    return problems


def shell_syntax_errors(path: pathlib.Path) -> list[str]:
    """`bash -n` over every block-scalar run body."""
    problems = []
    data = yaml.safe_load(path.read_text())
    for job_name, job in (data.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            script = step.get("run")
            if not script or "\n" not in script:
                continue
            proc = subprocess.run(["bash", "-n"], input=script,
                                  text=True, capture_output=True)
            if proc.returncode:
                problems.append(
                    f"{path.name} / {job_name} / "
                    f"{step.get('name', '<unnamed>')}: "
                    f"{proc.stderr.strip()}")
    return problems


def main() -> int:
    failures: list[str] = []
    checked = 0
    for path in sorted(WORKFLOWS.glob("*.yml")):
        checked += 1
        failures += folded_multiline_runs(path)
        failures += shell_syntax_errors(path)
    print(f"checked {checked} workflow files")
    for f in failures:
        print(f"  FAIL {f}")
    if not failures:
        print("  no folded multi-line `run:` scalars, no shell syntax errors")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
