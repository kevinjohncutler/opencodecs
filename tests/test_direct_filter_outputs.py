"""Independent bit streams and byte planes validate direct destinations."""
import struct

import numpy as np
import pytest

from opencodecs._packints_codec import PackintsCodec
from opencodecs._dicomrle_codec import DicomRleCodec


def packed(values, width):
    bits = ''.join(format(int(value), f'0{width}b') for value in values)
    bits += '0' * (-len(bits) % 8)
    return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))


@pytest.mark.parametrize('width', [1,2,3,4,7,9,10,12,14,17,24,31,33,40,48,56,63,64])
@pytest.mark.parametrize('order', ['<','>'])
def test_packed_destination(width, order):
    dtype = np.dtype(order + ('u1' if width <= 8 else 'u2' if width <= 16 else 'u4' if width <= 32 else 'u8'))
    values = [0,1,(1 << width)-1, (1 << width)//2,3 & ((1 << width)-1)]
    expected = np.asarray(values,dtype=dtype)
    out = np.empty_like(expected)
    codec = PackintsCodec()
    data = packed(values,width)
    assert codec.decode(data,dtype=dtype,bitspersample=width,n_elements=5,out=out) is out
    np.testing.assert_array_equal(out,expected)
    np.testing.assert_array_equal(codec.decode(data,dtype=dtype,bitspersample=width,n_elements=5),expected)
    with pytest.raises(ValueError):
        codec.decode(data[:-1],dtype=dtype,bitspersample=width,n_elements=5)


def test_packed_invalid_destinations_and_width():
    codec=PackintsCodec()
    for width in (0,65,-1):
        with pytest.raises(ValueError):codec.decode(b'123',dtype='u1',bitspersample=width)
    out=np.zeros(8,dtype='u1');out.flags.writeable=False
    with pytest.raises(ValueError):codec.decode(b'123',dtype='u1',bitspersample=3,out=out)
    with pytest.raises(ValueError):codec.decode(b'123',dtype='u1',bitspersample=3,shape=(7,),n_elements=8)
    target=np.zeros(16,dtype='u1')[::2]
    assert codec.decode(b'123',dtype='u1',bitspersample=3,out=target) is target


def literal_planes(array):
    planes=[]
    channels=array.shape[2] if array.ndim==3 else 1
    for ch in range(channels):
        values=array[...,ch] if array.ndim==3 else array
        for shift in reversed(range(array.dtype.itemsize)):
            raw=bytes((int(value) >> (shift*8)) & 255 for value in values.flat)
            planes.append(bytes([len(raw)-1])+raw)
    offsets=[];offset=64
    for plane in planes:offsets.append(offset);offset+=len(plane)
    return struct.pack('<16I',len(planes),*offsets,*([0]*(15-len(planes))))+b''.join(planes)


@pytest.mark.parametrize('dtype',['u1','i1','<u2','>u2','<i2','>i2'])
@pytest.mark.parametrize('shape',[(3,5),(3,5,3)])
def test_dicom_byte_plane_destination(dtype,shape):
    expected=(np.arange(np.prod(shape))*257).astype(dtype).reshape(shape)
    blob=literal_planes(expected)
    codec=DicomRleCodec()
    out=np.empty_like(expected)
    assert codec.decode(blob,shape=shape,dtype=dtype,out=out) is out
    np.testing.assert_array_equal(out,expected)
    np.testing.assert_array_equal(codec.decode(codec.encode(expected),shape=shape,dtype=dtype),expected)
    backing=np.empty((shape[0]*2,*shape[1:]),dtype=dtype)
    sliced=backing[::2]
    codec.decode(blob,shape=shape,dtype=dtype,out=sliced)
    np.testing.assert_array_equal(sliced,expected)
    out.flags.writeable=False
    with pytest.raises(ValueError):codec.decode(blob,shape=shape,dtype=dtype,out=out)


@pytest.mark.parametrize('dtype',['u1','i1','<u2','>u2','<i2','>i2','<u4','>u4','<i4','>i4'])
def test_rice_owned_destination(dtype):
    pytest.importorskip('opencodecs.codecs._rcomp')
    from opencodecs._rcomp_codec import RcompCodec
    codec=RcompCodec()
    expected=np.arange(-64,64).astype(dtype).reshape(8,16)
    blob=codec.encode(expected)
    out=np.empty_like(expected)
    assert codec.decode(blob,dtype=dtype,shape=expected.shape,out=out) is out
    np.testing.assert_array_equal(out,expected)
    np.testing.assert_array_equal(codec.decode(blob,dtype=dtype,shape=expected.shape),expected)
    out.flags.writeable=False
    with pytest.raises(ValueError):codec.decode(blob,dtype=dtype,shape=expected.shape,out=out)


def test_rice_header_validation():
    native=pytest.importorskip('opencodecs.codecs._rcomp')
    for nbytes,block,bpp in [(10,32,0),(10,32,3),(3,32,2),(8,0,2),(8,2**31,2)]:
        with pytest.raises((ValueError,RuntimeError)):
            native.decode(struct.pack('<IIi',nbytes,block,bpp)+bytes(16))
