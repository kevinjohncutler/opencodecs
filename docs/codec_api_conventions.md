# Codec API conventions

How a codec class should look — what kwargs to accept, which methods
to implement, what they should return. Pulled out of the codebase
after an audit that tightened up 4+ duplicated coercion helpers and
fixed a `mode=` / `backend=` naming inconsistency between VSI and
the other hybrid codecs.

The goals: a user who learned ND2 should be able to use LIF without
re-reading the docs, and a maintainer adding the next vendor format
should know exactly which slots to fill.

## The Codec class

Every registered codec subclasses `opencodecs.core.codec.Codec` and
exposes a stable handful of methods.

```python
class FooCodec(Codec):
    name = "foo"
    file_extensions = (".foo",)
    aliases = ()

    has_native = True            # we have a built-in implementation
    has_delegate = _HAVE_FOO     # an external package can serve as fallback
    can_encode = False
    can_decode = True
    multi_frame = True
    chunked = True               # supports random-access per frame
    streaming_decode = True      # opens before the full file is read
    parallel_decode = False

    supported_dtypes = (np.uint8, np.uint16, np.float32)
    supports_color = True

    def signature(self, head: bytes) -> bool: ...
    def decode(self, src: Any, **opts) -> np.ndarray: ...
    def encode(self, arr, **opts) -> bytes: ...           # if can_encode
    def open(self, src: Any, **opts) -> Reader: ...       # if multi_frame
    def info(self, src: Any) -> dict: ...                 # if partial-parseable
```

Don't invent additional public methods — anything else belongs on the
`Reader` the codec hands out.

## `signature(head)`

Receives the first ~16 bytes of a file and returns `True` if this
codec recognises the magic. Used by `oc.codec_for_bytes`. Keep this
cheap and side-effect-free — it's tried against every registered
codec at dispatch time.

## `decode(src)` / `encode(arr)`

Bytes-in / bytes-out. The minimum interface; even container formats
support this for the convenience of users who don't care about
streaming.

## `open(src)` and the `backend=` kwarg

For hybrid codecs (native + delegate), `open()` accepts a
`backend=` keyword with three values:

```python
def open(self, src: Any, *, backend: str | None = None, **opts) -> Reader:
```

* `backend=None` (default): try the native path; fall back to the
  delegate on `NotImplementedError` / supportable failure modes.
* `backend="native"`: force the native parser. Raise if it can't
  handle the input.
* `backend="<delegate-name>"`: force the delegate (e.g. `"nd2"`,
  `"readlif"`, `"oiffile"`).

**Use `backend=` even when the choice isn't strictly native-vs-delegate.**
VSI's three options (`auto` / `thumbnail` / `ets`) are both native code
paths, but we still call the kwarg `backend=` for cross-codec
consistency. The actual semantics live in the docstring.

If you're keeping an older kwarg around for back-compat, accept it as
an alias:

```python
def open(self, src, *, backend="auto", mode=None, **opts):
    if mode is not None:
        backend = mode
    ...
```

## `info(src)` — partial parse without decoding

Returns a dict describing the file's structure without touching pixel
data. The implemented codecs use this for OIR / VSI / ND2 / LIF / OIB /
LERC. Convention:

* Accept the same input types as `open()`: path, `DataSource`,
  bytes, file-like.
* Read only the metadata regions (header, directory, attributes
  chunk, OLE2 streams). For HTTP-backed sources this should cost a
  handful of `read_at` calls, not the full file.
* Return a flat-ish dict. Common keys: `file_size`, `n_frames`,
  `shape`, `dtype`, plus codec-specific geometry. Nest under a
  `layout`/`stack`/`images` key when there's per-element detail.

## Accepting a path or a `DataSource`

Every native reader takes a path *or* a `DataSource` (FileDataSource
for local, HTTPDataSource for remote, custom backends for S3 / DICOMweb /
etc.). The shared helper does the coercion + size probe:

```python
from .core.io import coerce_data_source

class FooNativeReader(Reader):
    def __init__(self, src: Any):
        self._src, self._owns_src, self._size = coerce_data_source(src)
        # self._size is populated even for HTTPDataSource (forces a
        # 4-byte primer read to get Content-Range).
        ...

    def close(self) -> None:
        if self._owns_src:
            try:
                self._src.close()
            except Exception:
                pass
```

Do **not** re-implement the path-vs-DataSource branching. The helper
is the right place to add new source types later (S3, GCS, in-memory
buffer with size hints) without touching every reader.

Bytes / file-like inputs go through `Codec.open()`'s pre-amble (spill
to temp file, then hand to the native reader). Don't push that branch
into the native reader.

## Lifecycle

* `Reader.__enter__` returns self; `__exit__` calls `close()`.
* `close()` is idempotent and only frees resources the reader owns
  (FDs it opened, mmaps it created, thread pools it spawned). If the
  caller passed in a DataSource, the caller owns its lifecycle.
* No `gc.collect()` in `close()` unless a `BufferError` raised by
  `mmap.close()` proves a memoryview leak. Unconditional `gc.collect()`
  hides bugs (e.g. the LERC use-after-free originally surfaced as a
  flaky `GC during CziReader.close` crash because of an unconditional
  collect masking the real corruption).

## Where the bodies are buried

* `coerce_data_source` — `src/opencodecs/core/io.py`
* `Codec` / `Reader` ABCs — `src/opencodecs/core/codec.py`
* Reference hybrid codecs — `_nd2_codec.py`, `_lif_codec.py`,
  `_oib_codec.py` (backend kwarg + delegate fallback)
* Reference native-only codecs — `_oir_codec.py`, `_vsi_codec.py`
  (full info() implementation)
* Reference small-blob codec — `_lz4.pyx`, `_qoi.pyx` (the
  `PyBytes_FromStringAndSize` + slice idiom for bytes outputs — see
  `_lerc.pyx` for the use-after-free pattern to avoid)
