"""Owned native transform lifetime and caller destination contracts."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest

from opencodecs._cms_codec import CmsTransform, _builtin_profile_icc, cms_transform


@pytest.fixture
def profile():
    try:
        return _builtin_profile_icc('srgb')
    except ImportError:
        pytest.skip('Little-CMS is unavailable')


@pytest.mark.parametrize('dtype', ['u1', 'u2'])
@pytest.mark.parametrize('channels', [3, 4])
def test_reusable_context_and_independent_lifetime(profile, dtype, channels):
    a = np.arange(11 * 17 * channels, dtype=dtype).reshape(11, 17, channels)
    expected = cms_transform(a, profile_in=profile)
    with CmsTransform(a, profile_in=profile) as transform:
        out = np.empty_like(a)
        assert transform(a, out=out) is out
        np.testing.assert_array_equal(out, expected)
        with ThreadPoolExecutor(3) as pool:
            for got in pool.map(transform, [a] * 8):
                np.testing.assert_array_equal(got, expected)
        out.flags.writeable = False
        with pytest.raises(ValueError, match='writable'):
            transform(a, out=out)
        with pytest.raises(ValueError, match='layout'):
            transform(a.astype('u2' if dtype == 'u1' else 'u1'))
    transform.close()
    with pytest.raises(ValueError, match='closed'):
        transform(a)


def test_context_invalid_profiles(profile):
    a = np.zeros((2, 3, 3), dtype='u1')
    with pytest.raises(ValueError, match='profile'):
        CmsTransform(a, profile_in=b'bad')
    with pytest.raises(ValueError, match='profile'):
        CmsTransform(a, profile_in=profile, profile_out=b'bad')
