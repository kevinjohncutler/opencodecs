"""Prepared segment encoders retain dynamic worker budgets and fidelity options."""
import pytest
from opencodecs.core import segment_compression as segments
from opencodecs.core.pipeline import WorkerBudget


@pytest.mark.parametrize("codec,inner", [(segments.ZSTD, 0), (segments.JXL, 1), (segments.JPEG2000, 1)])
def test_bound_encoder_preserves_eager_threads_and_caps_inner_work(monkeypatch, codec, inner):
    calls, lookups = [], []
    def encoder(data, **options):
        calls.append(options)
        return data
    def lookup(code, side):
        lookups.append((code, side))
        return encoder
    monkeypatch.setattr(segments, "_lookup_fn", lookup)
    bound = segments.bind_segment_encoder(codec, level=4, numthreads=7,
                                          owned_output=True, lossless=True)
    assert bound(b"first") == b"first"
    assert WorkerBudget(2).run(bound, b"nested") == b"nested"
    assert bound(b"last") == b"last"
    assert [options["numthreads"] for options in calls] == [7, inner, 7]
    # JPEG XL's encoder takes imagecodecs' level as its quality.
    key = "quality" if codec == segments.JXL else "level"
    assert all(options[key] == 4 and options["lossless"] is True for options in calls)
    assert lookups == [(codec, "encode_buffer" if codec == segments.ZSTD else "encode")]
