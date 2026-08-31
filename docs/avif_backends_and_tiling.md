# AVIF: encoder backends and tile-parallel encoding

Findings behind the `codec=`, `tile_cols_log2=` / `tile_rows_log2=`,
`auto_tiling=`, `yuv_format=` and `codec_options=` kwargs on
`opencodecs.codecs._avif.encode`.

All numbers below are Apple Silicon (M-series, macOS), libavif 1.3.0,
libaom 3.13.0, SVT-AV1 3.1.2, on a synthetic 2048x2048 RGB frame with
photo-like statistics at `level=80`. Reproduce with
`tests/test_heif_avif_features.py` plus your own timing loop; treat the
absolute milliseconds as machine-specific and the ratios as the finding.

## Tile-parallel encoding is the win

AV1 tiles are encoded by independent threads, so splitting a large frame
costs a little compression efficiency and buys real wall-clock time.

| Tiles | Encode | Size | PSNR |
|---|---|---|---|
| 1 (untiled) | 186 ms | 50.5 KiB | 39.6 dB |
| 4 (2x2) | 109 ms | 51.1 KiB | 39.6 dB |
| 16 (4x4) | 92 ms | 52.7 KiB | 39.6 dB |
| 64 (8x8) | 90 ms | 54.9 KiB | 39.6 dB |

4x4 is about 2x faster than untiled for +4.4% bytes and no measurable
quality change. 8x8 buys nothing further and costs +8.7%, so 4x4 is the
ceiling. `encode()` therefore defaults to `log2=2` on both axes once the
long axis reaches 1024 px, and to no tiling below that, where the
per-tile header overhead would dominate.

## SVT-AV1 is an alternative backend, not a faster one

SVT-AV1 is tuned for video sequences: long GOPs, motion estimation, and
rate control across frames. Driven by libavif for a single still it
allocates rate as though the image were one I-frame of a stream, and it
logs `Only a single picture was passed in` because libavif does not put
it in single-image mode.

| speed | libaom | SVT-AV1 |
|---|---|---|
| 6 | 82 ms / 52.7 KiB | 609 ms / 86.2 KiB |
| 10 | 34 ms / 77.8 KiB | 262 ms / 658.8 KiB |

libaom stays the default. `codec='svt'` remains exposed for callers who
specifically want it, and it raises `AvifError` when libavif was built
without `AVIF_CODEC_SVT`. Revisit if libavif starts setting SVT's
single-image flag.

## codec_options is an escape hatch, not a tuning recipe

Keys go straight to the backend and must match its own spelling. For
libaom, `enable-cdef`, `enable-restoration`, `row-mt`, `aq-mode`,
`tune`, `sharpness`, `enable-tpl-model` and `deltaq-mode` are all
accepted; note the post-process toggle is `enable-cdef`, not `cdef`,
which raises `AvifError`. Disabling the post-process filters measured
0.83x (that is, slower) at `speed=10` with byte-identical output, so
benchmark before reaching for any of these.

## Chroma layout

`yuv_format` selects `'420'` (default for lossy), `'422'`, `'444'` or
`'400'`. Lossless always forces 4:4:4, since chroma subsampling is lossy
by definition. 4:2:0 is right for natural-image content, but on
sparse-bright or synthetic content with saturated adjacent hues it
bleeds badly: on alternating red/blue columns, 4:4:4 measured 52.9 dB
against 4:2:0's 7.6 dB.

## Known build wart: link order on dev machines

On a macOS dev box with Homebrew's libavif installed alongside the
source build from `bench/build_codec_libs.sh`, the extension can link
against the wrong one. `setup.py` correctly puts
`$OPENCODECS_CODEC_LIBS_PREFIX/lib` first in the extension's
`library_dirs`, but a pyenv-built interpreter bakes `-L/opt/homebrew/lib`
into its `LDSHARED`, and those flags precede the extension's own `-L` in
the link command. The linker then resolves `-lavif` to the Homebrew copy
and bakes its absolute install name into the `.so`.

The symptom is silent: `codec='svt'` fails with `No codec available`
even though the source-built libavif has the SVT encoder, because the
Homebrew build does not. Confirm with
`otool -L src/opencodecs/codecs/_avif.*.so`; the correct result is
`@rpath/libavif.16.dylib`, not a `/opt/homebrew/...` path.

Workaround when building locally:

```sh
P="$OPENCODECS_CODEC_LIBS_PREFIX"
export LDSHARED="clang -bundle -undefined dynamic_lookup -L$P/lib -Wl,-rpath,$P/lib"
```

This does not affect the released wheels, which build in a container
with no Homebrew. A proper fix (linking the resolved library by absolute
path via `extra_objects`, or stripping the Homebrew `-L` from the build
environment) is still open, and it affects every codec that Homebrew
also ships, not just AVIF.

## Still open

1. Add `svtav1` to the `CIBW_BEFORE_ALL` codec-library builds and bump
   `CACHE_VERSION` so wheels link a libavif that has the SVT backend.
   Until then `codec='svt'` raises on wheel installs.
2. Fix the link-order wart above in `setup.py` rather than documenting
   a workaround.
3. Confirm the high-level `opencodecs.write(format='avif', ...)` path
   forwards the new kwargs through `**opts`.
