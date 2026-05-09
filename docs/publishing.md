# Publishing opencodecs to (Test)PyPI

The `build_wheels.yml` workflow can publish to TestPyPI or PyPI via
[Trusted Publishing][tp] (OpenID Connect, no API tokens). This doc
captures the one-time setup needed before the publish steps will run.

[tp]: https://docs.pypi.org/trusted-publishers/

## 1. One-time TestPyPI setup

The `publish_testpypi` job uses GitHub OIDC to obtain a short-lived
upload token. TestPyPI needs to know who's allowed to mint those tokens.

1. Create a TestPyPI account at <https://test.pypi.org/account/register/>
   (or sign in if you already have one).

2. Go to <https://test.pypi.org/manage/account/publishing/> and add a
   **pending publisher** for the project that doesn't exist yet:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `opencodecs` |
   | Owner | `kevinjohncutler` |
   | Repository name | `opencodecs` |
   | Workflow name | `build_wheels.yml` |
   | Environment name | `testpypi` |

3. (Optional but recommended) Repeat at
   <https://pypi.org/manage/account/publishing/> for the production
   PyPI project — same fields, environment name `pypi`.

## 2. GitHub deployment environments — already created

Two empty deployment environments named `testpypi` and `pypi` already
exist on the repo (created via the gh API during initial setup). The
workflow's `environment: { name: testpypi, ... }` references attach to
them automatically. Trusted Publishing handles auth via OIDC; the
environments don't need any secrets or variables.

If you ever want to add a manual approval gate before production
publishes, visit
<https://github.com/kevinjohncutler/opencodecs/settings/environments/pypi>
and add yourself as a Required reviewer. Without that, a tag push to
`v*` triggers a publish with no human in the loop.

To verify both environments still exist:

```bash
gh api repos/kevinjohncutler/opencodecs/environments --jq '.environments[].name'
# expects: testpypi
#          pypi
```

## 3. Trigger a TestPyPI publish

After the one-time setup:

```bash
gh workflow run build_wheels.yml -f publish=testpypi
```

Steps:

1. `build_wheels` matrix builds wheels for all OS / arch cells
   (Linux x86_64 + aarch64, Windows AMD64, macOS arm64) plus an
   sdist.
2. All wheels + the sdist land as artifacts on the workflow run.
3. `publish_testpypi` downloads them and pushes to TestPyPI via
   `pypa/gh-action-pypi-publish`.

A successful run shows up at
<https://test.pypi.org/project/opencodecs/>. Verify it installs:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            opencodecs
```

## 4. Production PyPI publish

Two ways:

```bash
# Manual one-shot:
gh workflow run build_wheels.yml -f publish=pypi

# Automatic on tag:
git tag v0.1.0 && git push origin v0.1.0
```

The tag-push path is the canonical release flow — `build_wheels.yml`
fires on `tags: ["v*"]`, the `publish_pypi` job auto-runs, and the
release is up on PyPI minutes later.

## Caveats

- **Mac wheels target macOS 15** because `macos-latest` is now
  Sequoia (15.x) and Homebrew formulas reference 15-only symbols.
  Older Macs get the sdist (which builds against whatever brew /
  conda libs they have).
- **Windows wheels need DLLs bundled** by `delvewheel` for true
  portability. The cibuildwheel default config does this; if a
  Windows wheel ends up < ~5 MB total something's off (the codec
  DLLs alone are ~10 MB compressed).
- **Linux wheels build inside the manylinux_2_28 container.**
  `bench/build_libjxl.sh` runs in `cibuildwheel`'s `BEFORE_ALL`
  hook to source-build libjxl 0.11+, since EPEL 8 ships 0.7. The
  `actions/cache`-backed mount at `/cibw-jxl-prefix` makes warm
  rebuilds ~30 s instead of ~5 min.
