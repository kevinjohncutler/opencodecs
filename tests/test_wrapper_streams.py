"""rcomp, aec, pcodec, sz3 and sperr write each library's own stream.

Up to 0.4.0 these five codecs put a private header in front of the
library's output, so nothing else could read what they wrote and they
could not read anything else wrote. Each now writes and reads the
stream its specification or library defines:

* rcomp: the bare cfitsio Rice stream, FITS RICE_1 (FITS 4.0, 10.4.1)
* aec: the bare CCSDS 121.0-B-2 coded stream libaec produces
* pcodec: pcodec's standalone format ('pco!')
* sz3: SZ3's own stream (header, payload, configuration)
* sperr: SPERR's own format (10-byte 2-D header; 3-D self-describing)

The references here are independent of opencodecs: blobs written by
imagecodecs 2026.8.16 (embedded below as hex, so these run without it),
fields placed by hand from the format documents, and, when imagecodecs
is installed, a live exchange with it in a separate process. Blobs
written by opencodecs 0.4.0 are embedded too, to pin that they still
open.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import pickle
import struct
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import opencodecs as oc
from _ic_reference import skip_if_old_imagecodecs  # noqa: E402

pytestmark = skip_if_old_imagecodecs


def _need(name):
    if not oc.has_codec(name):
        pytest.skip(f"codec {name!r} not built")


def _blob(table, key):
    return bytes.fromhex("".join(table[key]))


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# The arrays the fixtures below encode.
ARRAYS = {
    "rcomp_i2": lambda: (np.arange(-40, 40, dtype="i2") * 3).reshape(8, 10),
    "rcomp_i4": lambda: (np.arange(-30, 30, dtype="i4") * 1001).reshape(6, 10),
    "rcomp_u1": lambda: (np.arange(72) % 9 * 20).astype("u1"),
    "aec_u2": lambda: (np.arange(96) % 17 * 5).astype("u2"),
    "aec_i2": lambda: (np.arange(96) % 13 - 6).astype("i2"),
    "aec_u2_32_128": lambda: (np.arange(128) % 29 * 3).astype("u2"),
    "pcodec_f4": lambda: np.linspace(0, 1, 24, dtype="f4").reshape(2, 3, 4),
    "pcodec_i2": lambda: (np.arange(30) * 7 - 100).astype("i2").reshape(5, 6),
    "sz3_f4": lambda: np.sin(np.arange(60) / 5).reshape(6, 10).astype("f4"),
    "sz3_f8": lambda: np.sin(np.arange(60) / 5).reshape(3, 4, 5).astype("f8"),
    "sperr_f4_2d": lambda: np.sin(np.arange(16 * 12) / 7).reshape(12, 16).astype("f4"),
    "sperr_f8_3d": lambda: np.sin(np.arange(4 * 6 * 8) / 9).reshape(4, 6, 8).astype("f8"),
    "rcomp_i4_lookalike": lambda: _lookalike(),
    "aec_i1": lambda: np.array([-5, 3, -128, 127, 0, 1, -1] * 10, "i1"),
    "aec_u1_pad_rsi": lambda: (np.arange(100) * 37 % 251).astype("u1"),
    "aec_u4_16_64": lambda: (np.arange(64, dtype="u4") * 104729 % 70001).astype("u4"),
    "aec_i4_nopre": lambda: (np.arange(48) * 37 % 101 - 50).astype("i4"),
    "sz3_f8_unpred": lambda: _spiky(64).astype("f8"),
    "sz3_f4_unpred": lambda: _spiky(150).astype("f4"),
    "aec_i2_12": lambda: np.array([-2048, -1, 0, 1, 2047, -2] * 16, "i2"),
    "aec_i4_20": lambda: np.array([-524288, -1, 0, 1, 524287, -7] * 16, "i4"),
    "aec_i1_4": lambda: np.array([-8, -1, 0, 1, 7, -3] * 16, "i1"),
    "aec_0d_u1": lambda: np.array(7, "u1"),
    "aec_0d_u2": lambda: np.array(1234, "u2"),
    "sz3_f4_default": lambda: np.sin(np.arange(60) / 5).reshape(6, 10).astype("f4"),
    "sperr_inf_q": lambda: np.random.default_rng(3).normal(0, 1e200, (8, 8)),
    "sperr_inf_q_3d": lambda: np.random.default_rng(3).normal(0, 1e200, (2, 4, 4)),
}


def _spiky(n):
    x = np.arange(n)
    return np.sin(x / 5.0) + np.where(x % 13 == 0, 1e3, 0)


# An int32 ramp whose first value, 262144, is 1024 (its byte count) when
# its big-endian bytes are read little-endian, and whose next values
# make the following eight bytes read as block size 208 and 4 bytes per
# pixel: its Rice stream starts with something that passes for the
# 0.4.0 header. The steps after the second value, as int8.
_LOOKALIKE_STEPS = (
    "223c3af32fc5face1e372cd229e330ecca39c7063ae6f7e4d0d5de3df42237cc"
    "033a2235fef832cbe7050de013f0c10d2acc150ac515dad126342b29f700f6e1"
    "d8d538131bd5ea3fc8ee0ddc05f119ebe4c4c905e6ef12ef3a0326282bd019f3"
    "e1e9fa0cd4c5d4d83304dbd03503212321dad81fd8e53f0205142bde100bf808"
    "3f39e7dcd33d1af8ef0bdf052bd6f40f06181cef2ddd01f309d2f61a341f1af3"
    "39ca3b170ffbd42d1e27fef3cdf730f4013204de26e12420041b1ccaebfadb33"
    "131ac0f8e02c3827322edd0f3914e4f022f4233f23122bf2c22bd50d1e1b0f17"
    "def6d231fbe6c71edfeff8ee0407fbd80d1d1638f737c5d9dc1b0de207c1"
)


def _lookalike():
    steps = np.frombuffer(bytes.fromhex(_LOOKALIKE_STEPS), "i1").astype("i8")
    a = np.empty(256, "i8")
    a[0], a[1] = 262144, 1074003976
    a[2:] = a[1] + np.cumsum(steps)
    return a.astype("i4")

# Written by imagecodecs 2026.8.16:
#   rcomp_encode(a) / rcomp_encode(a, nblock=16) ("rcomp_i2_16")
#   aec_encode(a) / aec_encode(a, blocksize=32, rsi=128)
#   aec_encode(a, flags=AEC_DATA_PREPROCESS | AEC_PAD_RSI, blocksize=8,
#              rsi=2, bitspersample=8) ("aec_u1_pad_rsi", by a libaec
#              built with ENABLE_RSI_PADDING)
#   aec_encode(a, flags=AEC_DATA_PREPROCESS | AEC_DATA_MSB) ("aec_u2_msb",
#              of ARRAYS["aec_u2"]; flags 12, since imagecodecs' AEC.FLAG
#              does not name the MSB bit)
#   aec_encode(a.astype(">i2"), flags=AEC_DATA_PREPROCESS)
#              ("aec_i2_be_flags8", of ARRAYS["aec_i2"]; imagecodecs sets
#              no signed flag for a big-endian int16 array)
#   rcomp_encode(a.astype(">i2")) ("rcomp_be_i2", of ARRAYS["rcomp_i2"])
#   sperr_encode(np.full((5, 7), 2.5, "f4"), 1e-3, "pwe") ("sperr_const_2d")
#   sperr_encode(np.full((3, 4, 5), -1.25, "f8"), 1e-3, "pwe")
#              ("sperr_const_3d")
#   pcodec_encode(a)
#   sz3_encode(a, mode="abs", abs=1e-3)
#   sperr_encode(a, 1e-3, "pwe")
#   aec_encode(a & 0xFFF, bitspersample=12) ("aec_i2_12"),
#   aec_encode(a & 0xFFFFF, bitspersample=20) ("aec_i4_20") and
#   aec_encode(a & 0xF, bitspersample=4, flags=AEC_DATA_PREPROCESS |
#              AEC_DATA_SIGNED) ("aec_i1_4"): the signed samples masked
#              to bits_per_sample bits, the form libaec takes; each
#              decodes in imagecodecs to the unmasked array
#   aec_encode(a) of 0-d arrays ("aec_0d_u1", "aec_0d_u2")
#   sz3_encode(a) with imagecodecs' defaults ("sz3_f4_default")
#   sperr_encode(a, 60, "psnr") ("sperr_inf_q", "sperr_inf_q_3d"):
#              SPERR's quantization step is infinite for these values,
#              and imagecodecs decodes the streams to all NaN
# Written by opencodecs 0.4.0 with each codec's defaults (aec then used
# block 32, RSI 128; sz3 abs_err=1e-3; sperr mode="pwe", pwe=1e-3),
# except "aec_u4_16_64" (block_size=16, rsi=64), "aec_i4_nopre"
# (preprocess=False) and "sperr_inf_q" (mode="psnr", psnr=60).
# IC_DECODED_SHA256 is the SHA-256 of imagecodecs' own decode of the
# lossy blobs: decoding the same stream must give the same values.
IC_BLOBS = {
    "rcomp_i2": (
        "ff8838ccccccccccccccccccccccccccccccc6cccccccccccccccccccccccccc"
        "cccccc6cccccccccccccccc0"
    ),
    "rcomp_i2_16": (
        "ff8838ccccccccccccccc6cccccccccccccccc6cccccccccccccccc6cccccccc"
        "cccccccc6cccccccccccccccc0"
    ),
    "rcomp_i4": (
        "ffff8ab25c007d27d27d27d27d27d27d27d27d27d27d27d27d27d27d27d27d27"
        "d27d27d27d27d27d27d27d27d27d27d27d27d27d25be93e93e93e93e93e93e93"
        "e93e93e93e93e93e93e93e93e93e93e93e93e93e93e93e93e93e93e93e93e900"
    ),
    "rcomp_u1": (
        "00d02850a142850a140102850a142850a140102850a142850a140102850a1465"
        "0a142802050a142850a142802050a142850a142802050a142850a1428020ca14"
        "2850a1428500"
    ),
    "aec_u2": (
        "30000492496aa8c924926aaa94014001ffd6aaaa97feaaaaaaa94012e00febd6"
        "aaa97feaaaaaaa94011b007eabd6aa97feaaaaaaa940107803eaabd6a97feaaa"
        "aaaa9400f3c01eaaabd697feaaaaaaa8"
    ),
    "aec_i2": (
        "1fffa249249fc1eaae8ffff7fd554783faeaa30000fc1aaacffeaaa8c000f83e"
        "aea3ffaaaa3000607feaa8ff07555c7fff9feaaa3e0fd5d500"
    ),
    "aec_u2_32_128": (
        "40000fffffff00000ef6db6db6db6db6db6db6dde4ffffffc00003fdb6db6db6"
        "db6db6db6dbbdb64fffffe00001ffdb6db6db6db6db6db77b6db64fffff00000"
        "fffdb6db6db6db6db6ef6db6db60"
    ),
    "pcodec_f4": (
        "70636f2103050406040105170000326421d31b01018000000040000200ffffff"
        "7f01000000805fbbe700"
    ),
    "pcodec_i2": (
        "70636f21030884070401081d0000e1001001010000c01002000600400300001d"
        "090aaaaaaa4a5555550000"
    ),
    "sz3_f4": (
        "10f342f3000203030101000000000000f00000000000000028b52ffd20f08107"
        "0000000000f96f4b3ed761c73e698c103fa6a4373fa46a573f1d9a6e3f6f467c"
        "3f0ee47f3f144e793fb7c7683f9ff94e3f28eb2c3fe6f7033f9183ab3ec38110"
        "3eba196fbd49d682be0a92e2beb8a21cbfcfbd41bf971f5fbf329c73bf89627e"
        "bfa5047fbf107c75bf162a62bfe5d345bfb19a21bf56e0edbe8c0f8fbec72aaa"
        "bd59b1ee3d59829f3e6dfcfc3e4630283fd12d4b3fbc11663f95c9773f89a07f"
        "3f95467d3fb8d3703ffec65a3f73013c3f22bd153f3201d33e413d643ed5f5ca"
        "3cb88232be27a3bbbef8440bbffd2a33bf6fec53bf033b6cbfa31e7bbf5cff7f"
        "bf64ab7abf1d596bbfe5a452bfdc8a31bf210204a63c000000000000000400fc"
        "a9f1d24d62503fa000000001001000000002"
    ),
    "sz3_f8": (
        "10f342f3000203039a01000000000000bc0200000000000028b52ffd60bc0145"
        "0c00b615493dd0dca63910407271c1d6e294802f13005ad503c3f160ab07be70"
        "ef80887217ae1ac4a46887a513dda70c229a4c07252c2a558e1cc23281d5bda5"
        "dc3b0537003400350077b8b1e1f0863564d8c21426a41e6107a6060c169c6003"
        "05123802507b2ec39edc3c8db01ee99291632788cc1334994e5799d851b5af0b"
        "2a5aec1a27e7107bf82042e7d62bd6175655c0f0d30f20a024d45cfa0082a403"
        "02dc0484440adef9e5e215a33861e212931831888fe7c6ce4cea67b3f4a2ac89"
        "c1f1e6b41e67605c961d92331b56de445bced918283f295c82b6ecb4ec768c4d"
        "e97cc6f8058c599c223f35564d4634562f2c3f3a86f838e9990174854402f225"
        "08111e457a808454f142a5154ac2c474d571c6300c735d7e3e5c985d9abfd66a"
        "66677e4a7efc6e961c30200003aa52c30118415c1c6eec04644801ad803bf00c"
        "ba06571146bb6553122d6865e1c320699d91167c343a90f5b8465f325d47b430"
        "452164341dcab144a6ae16058fb252d958be31b337ffb433f0378b548baafd76"
        "50fe61a0058c0d3c042c22030363013c000000000000000200fca9f1d24d6250"
        "3fa000000001000600000003"
    ),
    "sperr_f4_2d": (
        "0020100000000c00000080555599a8b0fead3ffa7e6abc7493583f0bc8050000"
        "00000000f25a317fc92f8edaa22e86c2d7f9c24b837c1e0000a08a3f0f2d2e8a"
        "6ba9075286c012bdbeb0ac8f954fb80e009ef4b6dc8439caccf99191011a4406"
        "1de2f1baf6e7065b03c241924130825c9cf01b8095a013030a630055082b1952"
        "3f7cfc249fa8e79e94f9670810890d400df40bc48e428971a7879ffea8b5ee00"
        "0118801055880627b0ce92da7d0f7a64d31c563148b41100a0889322fb14d2b6"
        "cb4d556eb56b09a0dbcc000000badc81659e08e41bb40de1ade65a24c0011800"
        "000000000000fac30b"
    ),
    "sperr_f8_3d": (
        "00400800000006000000040000002501000080d1867c78f510b53ffa7e6abc74"
        "93583f0a5208000000000000cbffbfe0cfe7e1ffacf5f930955a8bb5ae5a2195"
        "5a73b1a06a15f3fd88fffffb90cf1ffcc0a8f2ffaa027f7ef1ff5fcea7526b7d"
        "3ef86bbe3c1f00000000000000000000f8872af85155f0e39f7000c6f98fc100"
        "1cc67f0c076030ff730cc0798a978b4bcd4f658b6755cbcf4a359f556a7e1a55"
        "3c8a547e96b3f83c65f21f33d45f5ca600180505e094e9fc8f30f73f9972d4d0"
        "27fa18a5bdef18d27dcc44e077740933beb6b4f345bea67d810dc519c64c6459"
        "5c2555e72df63069ddcebcd93fb42f6e6b52d7811f79cf38fbe755c778901b70"
        "7f0450f99bcd2c9cdc6e57c7d8c45adb76133f0208b929e2d60440495fa9ad57"
        "f2e1dc598f4f3b7632ae416aa2e15cb2f74408381b8303"
    ),
    "rcomp_i4_lookalike": (
        "00040000d0000000040000008000000220000003c0000003a0000000c8000002"
        "f0000003a80000005800000318000001e000000370000002c0000002d8000002"
        "90000001c8000003000000013800000358000003900000038800000060000003"
        "a00000019800000088000001b8000002f8000002a800000218000003d0000000"
        "b80000022185c279868482a8ef120a58d5d3f4df8f7a3413aad06aa8acf4b050"
        "6c658c06cde979a8c133635563e17a3e89eaf59296e3716d4cd0a4423498b0c0"
        "d8fcccb97ab6be0dc6a6e5e26a0a4fc5531113111597be2f6a3e92a506c46836"
        "35f80f864c49ce474d2f43b10d46c67bfd6307084e8962e723bcda0a1f19a723"
        "2158d977d49b9d3c2e8f912e220de212504658f4a08286ce056a6b3a4b366e8f"
        "f3ffac3827322e22de72d1defa257467e46c8ad6ded5956d7cec6f9711e67622"
        "a598c5e10a1bd1d17525fd3a5830c45c352d276dd3bb87a0"
    ),
    "aec_i1": (
        "ff7f90003fa03fe09ff20007f407fc13f3c000fe80ff827e40701fd01ff04fc8"
        "001e0003fe09f90003fbc07fc13f20007f407ff827e4000fe80fff04fc8001fd"
        "01ff04e070003fa03fe00000"
    ),
    "aec_u1_pad_rsi": (
        "e004a9494949495efcb5292929292bbcd4eb494949495ce7a95d2929292b7d15"
        "2928f6895f85094949495d2bd0c12929292928e26709494949495c9d01292929"
        "292b7120eda949495f2569495d29292bc4cd292928f8e95d27694949495d2b85"
        "0d2929292bf8a4c1087eaa800000"
    ),
    "sz3_f8_unpred": (
        "10f342f300020303b2010000000000001d0200000000000028b52ffd601d0105"
        "0d0084164000200000000100f43f004002fca9f1d24d62503f00800000180040"
        "8f402c2a352f2bd6bd3f322de83937e3adbfe9b615085750c6bf003e0c9a94e0"
        "efbfc542e0c781fcef3f9cd1c3f5649de53fcdfebfc294f4e63f1d24849c6039"
        "8f40c4551aa243d3ed3f775153afee388f40a99bd51da887cc3f524ff3f59e68"
        "e1bf8d1b7360d463efbf6b672be93aecd83fe45533a29c54eabf24d6d0828432"
        "c5bf21e82ebf1f448f403399e0f581afeebf700532977cbae8bfbcd23d23ff6d"
        "c93f2e204b04fd478f403f66d0995f65e6bfd0252c526087edbf002100007d0a"
        "00010003000506070800000b000d0000001112000015161700001a00001d001f"
        "000002000400100f0a0900000c000e000000141300001c191800001b00001e80"
        "004d800000b77f00fdaa830000f77f000012fa05800000022a82000003fa7f00"
        "0038fe7f00277e0000c3010101010100010140000337e63ff0f67873b97a7de7"
        "9c036a955a6bb4aad36beee71700c76b4550463a0394065a6456e19cd7b09780"
        "2ab7bde47793bd01f717d5ee9564a0dd6046e0209056333c80116000181ca010"
        "036d2101074040000000000000000200fca9f1d24d62503fa000000001008000"
        "000001"
    ),
    "sz3_f4_unpred": (
        "10f342f30002030367020000000000006a0300000000000028b52ffd606a02ad"
        "1200e6627f4590d84a6bbdfe60c53b358ca2f1b5724cad74bd9ab9887c8eed3d"
        "018f5f2d02da78d545d87cbdf56d94e4ced2e32c4dcb996d783320ca96201249"
        "c75ff6ad0547f9f2b03b057b006e005900405f0e55aa39a3160b172d6a31544d"
        "9963e956a2c9d9c1ec7882fdd038692a97f2c453542dba7dd1ae183e74c9ccc6"
        "e945942bdb922a25fe785b8362b36ab1888f12c905511b57534da714a78da9a1"
        "e6b523640d492c48b96388ddd3951e214aae1f9552ca0ca06a2efc3be560c22a"
        "09dd59de0bd454cbe41fbc2c3f14fd9d9a86aff449484443421082dec12f50f0"
        "3f8f8067e775fec6e65f811ff00758c61703f3423e13f6553f5b76b528474649"
        "0cd76a918d09fd20fc51b0281baac913578b4ea8d4d28667a50ca4ab138513cc"
        "9924d4a59a744a59e565774e29399e5fc65f35837ad534429216c47bf31768eb"
        "3e8ccdc8df23e3f18d3c17e6afd576eeaff02b63f5fefab6776fdbde02b943aa"
        "dfc2853faac24fe1a9fca59730e18d3e42d11193d4033d83069fe07b7c826fe0"
        "7172061e7802bf8001ccd0d43cc0f019fea91a38748ca8eb92fa5a82e7c30ada"
        "961b4fc84c8bb685a77785a546046baaf4d1813b8978f8e8f2846f5a3d20ac34"
        "3411a1f65c04e53872b5824c2aaa0067b244b0809165ef97facb48f24972f2ef"
        "6cfebebccddbfbebe16d9798376afd189c5df163735bb41763db387c238ed88b"
        "bf012920d08229ab3950908330f88285054ac0e578398680f37a169ead19038c"
        "19bd30e76b3b5e4099ae7197221d050c5c60141c656934a314f80c9ab13486b6"
        "bc79620cdb19e359aa1181d4768efb433e107351889936210108969600000000"
        "0000000200fca9f1d24d62503fa000000001008000000001"
    ),
    "aec_u2_msb": (
        "b000049249402008020080200802c924926008020080200802008035400001ff"
        "ffd4028028028028028037fe80280280280280280280280352c0200fe803ffd4"
        "028028028028037fe80280280280280280280280351803007e802803ffd40280"
        "28028037fe80280280280280280280280350403803e802802803ffd402802803"
        "7fe8028028028028028028028034f003c01e802802802803ffd4028037fe8028"
        "02802802802802802800"
    ),
    "rcomp_be_i2": (
        "88ffb800c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c0"
        "0c00c00c00c00c00c00c00c00c00c00c00c016c00c00c00c00c00c00c00c0080"
        "4c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00"
        "c00c016c00c00c00c00c00c00c00c00c00c00c00c00c00c00c00c000"
    ),
    "sperr_const_2d": (
        "002007000000050000008123000000000000000000000000000440"
    ),
    "sperr_const_3d": (
        "004005000000040000000300000011000000813c000000000000000000000000"
        "00f4bf"
    ),
    "aec_i2_be_flags8": (
        "efafff80c4002001000800401fff08077c0710008004002001006bfc400200ef"
        "dffc07c400200fff84004002001007603f10008035fe2001000800400200e000"
        "0fc042002001000800400200d7ff7c0710008004002001007ffc200200e0300e"
        "03c4002001006bfc4002001007603f1000803ffe1001000800400200f0600faf"
        "f02000200020002000200ffffefc0610010008004002001006bfc401df9ffc07"
        "8800400200fff8400400200ee03e20010008035fe2001000800400"
    ),
    "aec_i2_12": (
        "f8007ff002002ffc801ffb7fff002002ffc801ffb7ff002002f7ff801ffb7ff0"
        "02002ffc801fffb7ff002002ffc801ffb7fff000002ffc801ffb7ff002002fff"
        "c801ffb7ff002002ffc801f8007ff002002ffc801ffb7fff002002ffc801ffb7"
        "ff002002f7ff801ffb7ff002002ffc801fffb7ff002002ffc801ffb7fff00000"
        "2ffc801ffb7ff002002fffc801ffb7ff002002ffc801"
    ),
    "aec_i4_20": (
        "fc00003ffff80001000017fffe400037fff8bfffffc000080000bffff20001bf"
        "ffc5ffffc000080000befffff0000dfffe2ffffe0000400005ffff90000dffff"
        "f17ffff0000200002ffffc80006ffff17fffff80000000017fffe400037fff8b"
        "ffff80001000017fffff20001bfffc5ffffc000080000bffff20001bf00000ff"
        "ffe0000400005ffff90000dfffe2ffffff0000200002ffffc80006ffff17ffff"
        "0000200002fbffffc00037fff8bffff80001000017fffe400037ffffc5ffffc0"
        "00080000bffff20001bfffc5fffffe0000000005ffff90000dfffe2ffffe0000"
        "400005fffffc80006ffff17ffff0000200002ffffc800060"
    ),
    "aec_i1_4": (
        "f0e45952fc8b2a5c8bbd4b916579722ca97e05952e45f2a5c8b2bc391654bf22"
        "ca9722ef52e4595e5c8b2a5f81654b917ca9722ca0"
    ),
    "aec_0d_u1": (
        "0078"
    ),
    "aec_0d_u2": (
        "002694"
    ),
    "sz3_f4_default": (
        "10f342f3000203030101000000000000f00000000000000028b52ffd20f08107"
        "0000000000f96f4b3ed761c73e698c103fa6a4373fa46a573f1d9a6e3f6f467c"
        "3f0ee47f3f144e793fb7c7683f9ff94e3f28eb2c3fe6f7033f9183ab3ec38110"
        "3eba196fbd49d682be0a92e2beb8a21cbfcfbd41bf971f5fbf329c73bf89627e"
        "bfa5047fbf107c75bf162a62bfe5d345bfb19a21bf56e0edbe8c0f8fbec72aaa"
        "bd59b1ee3d59829f3e6dfcfc3e4630283fd12d4b3fbc11663f95c9773f89a07f"
        "3f95467d3fb8d3703ffec65a3f73013c3f22bd153f3201d33e413d643ed5f5ca"
        "3cb88232be27a3bbbef8440bbffd2a33bf6fec53bf033b6cbfa31e7bbf5cff7f"
        "bf64ab7abf1d596bbfe5a452bfdc8a31bf210204a63c00000000000000040000"
        "00000000000000a000000001001000000002"
    ),
    "sperr_inf_q": (
        "000008000000080000008015661f457e092be9000000000000f07f0000000000"
        "00000000"
    ),
    "sperr_inf_q_3d": (
        "00400400000004000000020000001a00000080fc86c0219daa17690000000000"
        "00f07f000000000000000000"
    ),
}

LEGACY_BLOBS = {
    "rcomp_i2": (
        "a00000002000000002000000ff8838ccccccccccccccccccccccccccccccc6cc"
        "cccccccccccccccccccccccccccccc6cccccccccccccccc0"
    ),
    "aec_u2": (
        "c000000000000000102080000800000050000ffff007fff5aaaaaaaaaaaaaaaf"
        "5aaaaaaaaaaaaa5c01ffff007ffeabd6aaaaaaaaaaaaaabd6aaaaaaaaaa97c01"
        "ffff007ffaaaaf5aaaaaaaaaaaaaaaf5aaaaaaaaa0"
    ),
    "aec_i2": (
        "c00000000000000010208000090000003fffafff07ffc1fd555557555555d547"
        "fc1fff07ffd555d555557555551c1fff07ffc1fbaaaaaaeaaaaabaa0"
    ),
    "pcodec_f4": (
        "50434f4f05030000020000000000000003000000000000000400000000000000"
        "0000000000000000000000000000000000000000000000000000000000000000"
        "000000000000000070636f2103050406040105170000326421d31b0101800000"
        "0040000200ffffff7f01000000805fbbe700"
    ),
    "sz3_f4": (
        "535a334f000200000a0000000000000006000000000000000000000000000000"
        "0000000000000000000000000000000010f342f3000203030101000000000000"
        "f00000000000000028b52ffd20f081070000000000f96f4b3ed761c73e698c10"
        "3fa6a4373fa46a573f1d9a6e3f6f467c3f0ee47f3f144e793fb7c7683f9ff94e"
        "3f28eb2c3fe6f7033f9183ab3ec381103eba196fbd49d682be0a92e2beb8a21c"
        "bfcfbd41bf971f5fbf329c73bf89627ebfa5047fbf107c75bf162a62bfe5d345"
        "bfb19a21bf56e0edbe8c0f8fbec72aaabd59b1ee3d59829f3e6dfcfc3e463028"
        "3fd12d4b3fbc11663f95c9773f89a07f3f95467d3fb8d3703ffec65a3f73013c"
        "3f22bd153f3201d33e413d643ed5f5ca3cb88232be27a3bbbef8440bbffd2a33"
        "bf6fec53bf033b6cbfa31e7bbf5cff7fbf64ab7abf1d596bbfe5a452bfdc8a31"
        "bf210204a63c000000000000000400fca9f1d24d62503fa00000000100100000"
        "0002"
    ),
    "sperr_f4_2d": (
        "535052520102000010000000000000000c000000000000000100000000000000"
        "0000000000000000000000000000000080555599a8b0fead3ffa7e6abc749358"
        "3f0bc805000000000000f25a317fc92f8edaa22e86c2d7f9c24b837c1e0000a0"
        "8a3f0f2d2e8a6ba9075286c012bdbeb0ac8f954fb80e009ef4b6dc8439caccf9"
        "9191011a44061de2f1baf6e7065b03c241924130825c9cf01b8095a013030a63"
        "0055082b19523f7cfc249fa8e79e94f9670810890d400df40bc48e428971a787"
        "9ffea8b5ee000118801055880627b0ce92da7d0f7a64d31c563148b41100a088"
        "9322fb14d2b6cb4d556eb56b09a0dbcc000000badc81659e08e41bb40de1ade6"
        "5a24c0011800000000000000fac30b"
    ),
    "sperr_f8_3d": (
        "5350525200030000080000000000000006000000000000000400000000000000"
        "0000000000000000000000000000000000400800000006000000040000002501"
        "000080d1867c78f510b53ffa7e6abc7493583f0a5208000000000000cbffbfe0"
        "cfe7e1ffacf5f930955a8bb5ae5a21955a73b1a06a15f3fd88fffffb90cf1ffc"
        "c0a8f2ffaa027f7ef1ff5fcea7526b7d3ef86bbe3c1f00000000000000000000"
        "f8872af85155f0e39f7000c6f98fc1001cc67f0c076030ff730cc0798a978b4b"
        "cd4f658b6755cbcf4a359f556a7e1a553c8a547e96b3f83c65f21f33d45f5ca6"
        "00180505e094e9fc8f30f73f9972d4d027fa18a5bdef18d27dcc44e077740933"
        "beb6b4f345bea67d810dc519c64c64595c2555e72df63069ddcebcd93fb42f6e"
        "6b52d7811f79cf38fbe755c778901b707f0450f99bcd2c9cdc6e57c7d8c45adb"
        "76133f0208b929e2d60440495fa9ad57f2e1dc598f4f3b7632ae416aa2e15cb2"
        "f74408381b8303"
    ),
    "aec_u4_16_64": (
        "000100000000000020104000080000008000000002492492492487a81ea04e44"
        "6979391161c4e4447693910d984e44255939105144e4604a5294a52900d24e47"
        "f241391f84e4e47d031391efca4e47ae21391e7464e478c11391825294a5294e"
        "f612723b5009c8eb1f2723a3f89c8e6dd272392f09c8e29b272381e89c8c1294"
        "a5294a6f2c9391b8704e46d0b9391afec4e46aea9391a7684e468c993919ee44"
        "e440"
    ),
    "aec_i4_nopre": (
        "c0000000000000002020800001000000fffffffe77ffffff98000000c7fffffe"
        "c7ffffffe800000117ffffff100000003800000167ffffff600000008ffffffe"
        "8fffffffb0000000dffffffed8000000000000012fffffff2800000050000001"
        "7fffffff78000000a7fffffea7ffffffc8000000f7fffffef000000018000001"
        "47ffffff400000006800000197ffffff97c0000005fffffff5ffffffff000000"
        "087ffffff8400000018000000afffffffac00000043ffffff43ffffffd400000"
        "06bffffff6bfffffffc00000093ffffff93ffffff93ffffff93ffffff93fffff"
        "f93ffffff93ffffff93ffffff93ffffff93ffffff93ffffff93ffffff93fffff"
        "f93ffffff93ffffff93ffffff93ffffff900"
    ),
    "sperr_inf_q": (
        "5350525200020000080000000000000008000000000000000100000000000000"
        "000000000000000000000000000000008015661f457e092be9000000000000f0"
        "7f000000000000000000"
    ),
}

IC_DECODED_SHA256 = {
    "sz3_f4": "d6bddefa6757f3d30a602ac9d6c4fdf305ebe3a032195003b089ee56e57f2f62",
    "sz3_f8": "1fc14abb8b0205c67635167f809a7700ce52ed13a8c245cfd73b0936fd63712a",
    "sperr_f4_2d": "2ab3ccd76c36d8303f80be26aef19a155562a0b807a76da58da78a3c97895d57",
    "sperr_f8_3d": "372721b2a5fa2ba025d863fe1762d63cc85572c7702c2e7bcf2d104aa34bb13d",
    "sz3_f8_unpred": "2d9dd9351a619cb326065fb9225392cf288dd680301e722afa7c0bd251557d21",
    "sz3_f4_unpred": "d04038cd1c19340f149eb321fa213a796f6fa1a26984e0507e45cfc36426571d",
}

# Written by SZ3 3.3.2's C++ API, SZ_compress<T>(conf, data, size), with
# conf.errorBoundMode = EB_ABS and conf.cmprAlgo set as named, from
# _cpp_data(dtype): layouts the C API (and so imagecodecs) never writes
# but other SZ3 users do.
#   regression_f4:      ALGO_LORENZO_REG, lorenzo + regression, eb 1e-3
#   composed_f8:        ALGO_LORENZO_REG, lorenzo + lorenzo2 + regression,
#                       eb 1e-3
#   nopred_f8:          ALGO_NOPRED, eb 1e-7
#   regression_only_f4: ALGO_LORENZO_REG, regression only, eb 1e-7
SZ3_CPP_BLOBS = {
    "regression_f4": (
        "10f342f300020303d000000000000000760300000000000028b52ffd607602f5"
        "05008406020002fca9f1d24d62403f008000d03e010000003f0000000300006e"
        "6402000000c6dc00000001014028000100010105fffffffffe503f0040220700"
        "002f9c00010203060504375f0000c05dba5e01881300007b023fffeffffbfffe"
        "ffbfff8fc7e3f8fc7e4924922b20108321ea1c1450136c07942067d3ab0c8f60"
        "bd51c66610c83d5332d8b8c1368e8db9f12122862201c280efa30201c268c4e2"
        "ec45c9c8367f7ec9b9925083f649e8151c11745e0ee360695e5944392c69c058"
        "22010d881388130000000000000000fca9f1d24d62503fa00000000100800000"
        "0001"
    ),
    "composed_f8": (
        "10f342f300020303eb00000000000000a40300000000000028b52ffd60a402cd"
        "060032481f2b6075d218e4278ac2d61444b92240db28621a84406b0874429261"
        "03cc7266609a0474856320bd453ed93205f5508e76135055492dd3e72a87520f"
        "56cdd74f63ac9e61e61198ccccd01ce33bd0dde29185b0a0706618ac0586b988"
        "fbc389c731f2b5ed6b99672bb6cc7f558293ca674756364e5f8e9b04927a7500"
        "e0303120608336ea2084749fa0b864c26f8029245880c08602ee9bd637de0333"
        "443ec38dc51eecd8e42c8588b039a11931660d0b403519146a40c9846d02b9dd"
        "68f5813508d6e28cc5185a4556cb00fb0c1f2659d9c66c9734382c22010d8813"
        "88130000000000000000fca9f1d24d62503fe000000001008000000001"
    ),
    "nopred_f8": (
        "10f342f300020303f701000000000000a09e00000000000028b52ffd60a09d2d"
        "0f00d6184f3a20af241d9b0cfd11615d8a0002d07992600019370198c2e20bc0"
        "0042a400841136fb9ae2a77eea6763dbb66ddb5ab66d7b63dd970f749e4f5a0a"
        "4d00300046002dfda16b680b74059a023d81964047a021d00f680774039a01bd"
        "80564027404033231303f3e2d2c2b2a292827252252321070c14203040400000"
        "a58b15fc1c3d2048b3e6030a16234a90b0810297ac2e595db2ba6475c9ea92d5"
        "fbbeeffbbeeffbbeeffbbeaf6ddb86773bf02e07deddc0bb1a783703ef62e0dd"
        "0bbc03b66ddbb66ddb262422201f1e9e0e8e0d0d2807e7c6ea92d525ab4b5697"
        "6a9a1999189817bbb8c52c5eb18a538ce213974d4844403e3c3c1d1c1b1ad027"
        "07e7e6445db2ba6415518c0846f4226a11a988524426a21024e44144a743d224"
        "21b2c34ec90cd25532126275c9ea92d525ab4b56effbbeeffbbeeffbbeeffbbe"
        "018145a831c82a859d39a10729a9943a12402080cc2790f2ffa7bc0a5903e900"
        "0d241db883ae4e589db0b74e08d94d159cd30fa000aadee1aacb85dac5cf3b8e"
        "f29f2b9bfce7803eebfb1596700298648990810ea901f221c7d4a2591752da04"
        "a7f87f28454bbca1c2fe5da8c93aa7e5b00ddb356956e57586031860610441f4"
        "f232efc021c6e354290d7a9d81f0e786ac79ce8e33c7d9d1ec38739c2428d0be"
        "3cc8719ada355522010d88138813000000000000030048afbc9af2d77a3ea000"
        "000001008000000001"
    ),
    "regression_only_f4": (
        "10f342f300020303ce03000000000000545200000000000028b52ffd605451e5"
        "1d009a48d00c3f60a64a1a03739db35131b042ee6928dee08afe503b6e003471"
        "e1e288cc5e392ba61b9e6e512e6f662b02809d112025e5dae00cdd02fa924056"
        "e551527fb4140a01aa009600a5852070c624466c947133269470866218d7b2f2"
        "ca1b39a6943dc2ba8c6a8ac048400f5a46d97163d4d1a04aafca1e21fd06a98c"
        "5a09c48a7c8a26d5969e8a827d548c55452b2495ae21640205da195d78412887"
        "5d9090705961c622814de58ba76389e218676c2364556ed961a364a47baf7b77"
        "83d4ccdc3a08028afec170d23c3755544fa0410d74dba35d688469208d3cc593"
        "466a1c1644fa27e7a47db0a4348d459df2b9ed51c1b52f6d23eb94efc917fd13"
        "274ca50842e9990f43f44fdf9eda79ad7c5ca0681fb23a2dfe30533d3a577a0c"
        "84523532c6f44e104aebbcbe29bbd2391e8cb48da099d6a93147970819d4334b"
        "c6f4b8f6a5644978573dda5b3a020dc915ed5420ed15af825e075ff9bc1e5fef"
        "bcd27995f30af8faf7eaf78ae6b5efb5f88af72ae6f5eed5cbab96d72baf73af"
        "515ee35e9dbcbebd2a79657b557b457b05f25af68ae355ecf5c5abd7ab8ad7f9"
        "aaf57af55af53af52af5faf40ae0f59f843f8f3f017ff6fdbcfb39f7f3ed27da"
        "4fb19ff3e7d4cf9fd8f744b427469d6d0579f8c4bff8468886646342c1c844e7"
        "c02835f5904cb6a4287e55e8327ec96043126a28439648888c672bcc4a6281b3"
        "bfc3d747af8c5e13bd1a7a1df40ae8d5cfeb9e573caf735ee1bcb6795df39ae6"
        "d5ccebbd8e790df30ae695cb6b9657bad729af72af70af4d5edd5edb5ed75ed3"
        "5ecf5e7dbc92bdd6788df13b7fb77ec9df12bf5abf227eb17e41fc5efd76f8b5"
        "fad5f05bf57be197ea97c2efd46f84df07bf0d7ea57e15fc22f885fa7dfa2df0"
        "ebf42be017c02fd3afd22fd2efaf8d5e17bd227a250c602d1c151e2a245298a7"
        "b050e1483dd886c3e170b49f6beca75b6989e93f0969f8707ed372ed89dd3a60"
        "af2c5eb95ec957ac57ab57aad707af085e9d5e957e2efa39f893ce4fbf9f783f"
        "b5fc8cfbc9f6b3eca7d7cfab9f4f4f3c3e71ee89f3b7d07ef597f85bfe9af825"
        "f13be237eb37c4ef875f0fbf1c7e37fc66f8c5f06be1b7c2ef845f09bf107e1d"
        "fc32f85df09be037eaf7c0af815f02bf037e03fc36fd2efd26052d20e0029270"
        "6ad2360fc53c14b6a6b0218a838395c039ceac137ef7222d4c786120d5857f68"
        "3105550e4605a92b32055556e08f8a515723ba877cf0772a66c99b14c9366505"
        "1514f4870ffe5ec5d291930ad01ecebf56d8e0a4699a8170d15c65093ef83f48"
        "3af4037f11c31c180c01a9339ebd03092b07a86b2b9563106e968e3ae0362201"
        "0d88138813000000000000000048afbc9af2d77a3e2000000001008000000001"
    ),
}


def _cpp_data(dtype):
    i = np.arange(5000)
    return ((i % 97) * 0.5 + (i // 997) * 0.25).astype(dtype)


# ---------------------------------------------------------------------------
# rcomp: FITS RICE_1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key,kw", [
    ("rcomp_i2", {}), ("rcomp_i2_16", {"blocksize": 16}),
    ("rcomp_i4", {}), ("rcomp_u1", {}),
])
def test_rcomp_writes_and_reads_imagecodecs_bytes(key, kw):
    _need("rcomp")
    a = ARRAYS[key.replace("_16", "")]()
    ref = _blob(IC_BLOBS, key)
    c = oc.get_codec("rcomp")
    assert bytes(c.encode(a, **kw)) == ref
    back = c.decode(ref, shape=a.shape, dtype=a.dtype, **kw)
    assert back.dtype == a.dtype and back.shape == a.shape
    np.testing.assert_array_equal(back, a)


def test_rcomp_constant_stream_by_hand():
    """cfitsio ricecomp.c writes the first pixel whole (big-endian),
    then per block an FS code of fsbits zeros when every difference is
    zero (4 bits for 16-bit pixels, 5 for 32-bit), padded to a byte."""
    _need("rcomp")
    c = oc.get_codec("rcomp")
    assert bytes(c.encode(np.full(32, 0x1234, "i2"))) == b"\x12\x34\x00"
    assert bytes(c.encode(np.full(64, 0x1234, "i2"))) == b"\x12\x34\x00"
    assert bytes(c.encode(np.full(32, 0x12345678, "i4"))) == b"\x12\x34\x56\x78\x00"
    back = c.decode(b"\x12\x34\x00", shape=(64,), dtype="i2")
    np.testing.assert_array_equal(back, np.full(64, 0x1234, "i2"))


def test_rcomp_nblock_is_an_alias_and_unknown_options_raise():
    _need("rcomp")
    a = ARRAYS["rcomp_i2"]()
    c = oc.get_codec("rcomp")
    assert c.encode(a, nblock=16) == c.encode(a, blocksize=16)
    np.testing.assert_array_equal(
        c.decode(_blob(IC_BLOBS, "rcomp_i2_16"), shape=a.shape, dtype=a.dtype,
                 nblock=16), a)
    with pytest.raises(ValueError):
        c.encode(a, nblock=16, blocksize=32)
    with pytest.raises(TypeError):
        c.encode(a, level=3)
    with pytest.raises(TypeError):
        c.decode(_blob(IC_BLOBS, "rcomp_i2"), shape=a.shape, dtype=a.dtype,
                 bitspixel=16)


def test_rcomp_bare_stream_needs_shape_and_dtype():
    _need("rcomp")
    with pytest.raises(ValueError, match="shape= and dtype="):
        oc.get_codec("rcomp").decode(_blob(IC_BLOBS, "rcomp_i2"))


def test_rcomp_reads_legacy_blobs():
    _need("rcomp")
    a = ARRAYS["rcomp_i2"]()
    old = _blob(LEGACY_BLOBS, "rcomp_i2")
    # The 0.4.0 layout: <IIi byte count, block size, bytes per pixel.
    assert struct.unpack_from("<IIi", old) == (a.nbytes, 32, 2)
    assert old[12:] == _blob(IC_BLOBS, "rcomp_i2")
    c = oc.get_codec("rcomp")
    np.testing.assert_array_equal(c.decode(old, shape=a.shape, dtype=a.dtype), a)
    # Without arguments, as 0.4.0 read it: flat, unsigned words.
    np.testing.assert_array_equal(c.decode(old), a.ravel().view("u2"))


def test_rcomp_out_and_byte_order():
    _need("rcomp")
    a = ARRAYS["rcomp_i2"]()
    ref = _blob(IC_BLOBS, "rcomp_i2")
    c = oc.get_codec("rcomp")
    out = np.empty_like(a)
    assert c.decode(ref, out=out) is out
    np.testing.assert_array_equal(out, a)
    be = c.decode(ref, shape=a.shape, dtype=">i2")
    assert be.dtype == np.dtype(">i2")
    np.testing.assert_array_equal(be, a)
    assert bytes(c.encode(a.astype(">i2"))) == ref


def test_rcomp_bare_stream_that_looks_like_an_old_blob():
    """A valid bare stream (imagecodecs writes the same bytes) whose
    first 12 bytes pass every check on the 0.4.0 header and whose rest
    even decodes as an old blob's payload. Only the bare reading
    encodes back to the input, so that is the one returned."""
    _need("rcomp")
    from opencodecs.codecs import _rcomp
    a = ARRAYS["rcomp_i4_lookalike"]()
    ref = _blob(IC_BLOBS, "rcomp_i4_lookalike")
    assert struct.unpack_from("<IIi", ref) == (a.nbytes, 208, 4)
    assert _rcomp.decode_framed(ref).size == 256     # the trap
    c = oc.get_codec("rcomp")
    assert bytes(c.encode(a)) == ref
    np.testing.assert_array_equal(c.decode(ref, shape=a.shape, dtype=a.dtype), a)
    out = np.empty_like(a)
    c.decode(ref, out=out)
    np.testing.assert_array_equal(out, a)
    # Without shape and dtype it is not taken for an old blob either.
    with pytest.raises(ValueError, match="shape= and dtype="):
        c.decode(ref)


# The Rice decoders read one byte at a time; a stream that ends early
# must stop them at its last byte. The child puts each stream against
# a page with no access rights, so any read past the end is a crash.
_RICE_GUARD = textwrap.dedent("""
    import ctypes, mmap
    import numpy as np
    from opencodecs.codecs import _rcomp
    page = mmap.PAGESIZE
    m = mmap.mmap(-1, 2 * page)
    mem = np.frombuffer(m, np.uint8)
    libc = ctypes.CDLL(None)
    libc.mprotect.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    assert libc.mprotect(mem.ctypes.data + page, page, 0) == 0

    def at_page_end(stream):
        view = mem[page - len(stream):page]
        view[:] = np.frombuffer(stream, np.uint8)
        return memoryview(view)

    cases = []
    # First pixel, then an FS code for ordinary Rice coding followed by
    # nothing but zero bits: the count of leading zeros never ends.
    cases.append((bytes.fromhex("00000007200000"), 4))
    cases.append((bytes.fromhex("0007400000"), 2))
    cases.append((bytes.fromhex("07400000"), 1))
    # Every truncation of real streams.
    rng = np.random.default_rng(3)
    for dt in ("i1", "i2", "i4"):
        a = np.cumsum(rng.integers(-40, 41, 300)).astype(dt)
        whole = bytes(_rcomp.encode(a))
        cases += [(whole[:k], a.itemsize) for k in range(len(whole))]
    for stream, bpp in cases:
        try:
            _rcomp.decode_raw(at_page_end(stream), nelements=300,
                              blocksize=32, bytes_per_pixel=bpp)
        except _rcomp.RcompError:
            continue
        raise SystemExit(f"decoded a stream that ends early: {stream[:8].hex()}")
    print("STAYED INSIDE", len(cases))
""")


@pytest.mark.skipif(os.name == "nt", reason="needs mprotect for a guard page")
def test_rcomp_decoder_stops_at_the_end_of_its_input():
    _need("rcomp")
    r = subprocess.run([sys.executable, "-c", _RICE_GUARD],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "STAYED INSIDE" in r.stdout, (
        r.returncode, r.stdout[-500:], r.stderr[-2000:])


# ---------------------------------------------------------------------------
# aec: CCSDS 121.0-B-2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key,kw", [
    ("aec_u2", {}), ("aec_i2", {}),
    ("aec_u2_32_128", {"block_size": 32, "rsi": 128}),
])
def test_aec_writes_and_reads_imagecodecs_bytes(key, kw):
    _need("aec")
    a = ARRAYS[key]()
    ref = _blob(IC_BLOBS, key)
    c = oc.get_codec("aec")
    assert bytes(c.encode(a, **kw)) == ref
    back = c.decode(ref, dtype=a.dtype, shape=a.shape, **kw)
    assert back.dtype == a.dtype and back.shape == a.shape
    np.testing.assert_array_equal(back, a)
    # imagecodecs' parameter names
    alias = {"blocksize": kw["block_size"], "rsi": kw["rsi"]} if kw else {}
    assert bytes(c.encode(a, bitspersample=16, **alias)) == ref
    out = np.empty_like(a)
    view = c.decode(ref, out=out, **alias)
    assert bytes(view) == a.tobytes()
    np.testing.assert_array_equal(out, a)


def test_aec_defaults_are_imagecodecs_and_flags_combine():
    """imagecodecs' defaults: block 8, RSI 2, preprocessing; a signed
    16-bit array adds the signed flag (AEC_DATA_SIGNED = 1)."""
    _need("aec")
    a = ARRAYS["aec_i2"]()
    c = oc.get_codec("aec")
    ref = _blob(IC_BLOBS, "aec_i2")
    assert bytes(c.encode(a, block_size=8, rsi=2, flags=8 | 1)) == ref
    assert bytes(c.encode(a, flags=8)) == ref         # signed still inferred
    assert bytes(c.encode(a.view("u2"), signed=True)) == ref
    assert bytes(c.encode(a, signed=False)) != ref
    with pytest.raises(ValueError):
        c.encode(a, block_size=16, blocksize=32)
    with pytest.raises(TypeError):
        c.encode(a, level=3)


def test_aec_reads_legacy_blobs():
    _need("aec")
    c = oc.get_codec("aec")
    for key in ("aec_u2", "aec_i2"):
        a = ARRAYS[key]()
        old = _blob(LEGACY_BLOBS, key)
        # The 0.4.0 preamble: <QBBHB3x size, bps, block, rsi, flags.
        size, bps, block, rsi, flags = struct.unpack_from("<QBBHB3x", old)
        assert (size, bps, block, rsi) == (a.nbytes, 16, 32, 128)
        assert bytes(c.decode(old)) == a.tobytes()
        np.testing.assert_array_equal(
            c.decode(old, dtype=a.dtype, shape=a.shape), a)


def test_aec_output_size_and_padding():
    _need("aec")
    c = oc.get_codec("aec")
    a = (np.arange(3001) % 50).astype("u2")
    blob = c.encode(a)
    # Unknown size: libaec pads the last 8-sample block, and the whole
    # stream decodes, as imagecodecs.aec_decode returns it.
    whole = c.decode(blob, bits_per_sample=16)
    assert len(whole) == 3008 * 2
    assert whole[:a.nbytes] == a.tobytes()
    np.testing.assert_array_equal(c.decode(blob, dtype="u2", shape=a.shape), a)
    # Trailing zero blocks are coded as "zero to the end of the
    # 64-block segment": still the exact size when it is known.
    z = np.zeros(200, "u2")
    z[:3] = (5, 9, 2)
    zb = c.encode(z, block_size=32, rsi=128)
    assert len(c.decode(zb, bits_per_sample=16, block_size=32, rsi=128)) == 2048 * 2
    np.testing.assert_array_equal(
        c.decode(zb, dtype="u2", shape=z.shape, block_size=32, rsi=128), z)
    # An output that cannot hold the data raises instead of truncating.
    data = bytes(range(256)) * 16
    eb = c.encode(data, bits_per_sample=8)
    with pytest.raises(RuntimeError):
        c.decode(eb, bits_per_sample=8, out=bytearray(100))
    with pytest.raises(RuntimeError):
        c.decode(eb, bits_per_sample=8, dtype="u1", shape=(len(data) + 64,))
    assert c.decode(eb, bits_per_sample=8, out=len(data)) == data


def test_aec_refuses_values_that_do_not_fit():
    _need("aec")
    c = oc.get_codec("aec")
    with pytest.raises(ValueError, match="do not fit"):
        c.encode(np.array([0, 4096], "u2"), bits_per_sample=12)
    with pytest.raises(ValueError, match="bytes"):
        c.encode(np.zeros(8, "u4"), bits_per_sample=12)
    a = np.array([0, 4095, 17, 2048] * 4, "u2")
    np.testing.assert_array_equal(
        c.decode(c.encode(a, bits_per_sample=12), bits_per_sample=12,
                 dtype="u2", shape=a.shape), a)


def test_aec_codes_values_not_memory_order():
    """A big-endian array is coded with AEC_DATA_MSB, so the stream
    holds the same sample values as for the native array."""
    _need("aec")
    c = oc.get_codec("aec")
    a = ARRAYS["aec_u2"]()
    ref = _blob(IC_BLOBS, "aec_u2")
    assert bytes(c.encode(a.astype(">u2"))) == ref
    out = np.empty(a.shape, ">u2")
    c.decode(ref, out=out)
    np.testing.assert_array_equal(out, a)
    np.testing.assert_array_equal(c.decode(ref, dtype=">u2", shape=a.shape), a)


def test_aec_explicit_flags_keep_the_byte_order_bit():
    """An explicit flags (or msb) sets AEC_DATA_MSB as given, as in
    imagecodecs: libaec reads the array's bytes in that order, and
    decoding returns the decoded bytes as they are, with or without
    dtype. Without either, the bit follows the array's byte order."""
    _need("aec")
    c = oc.get_codec("aec")
    a = ARRAYS["aec_u2"]()
    ref = _blob(IC_BLOBS, "aec_u2_msb")
    native = _blob(IC_BLOBS, "aec_u2")
    assert ref != native
    assert bytes(c.encode(a, flags=12)) == ref
    assert bytes(c.encode(a, msb=True)) == ref
    assert bytes(c.encode(a, bitspersample=16, flags=8 | 4)) == ref
    # imagecodecs writes the same bytes for the big-endian array with
    # the bit clear: libaec reads its bytes least significant first.
    assert bytes(c.encode(a.astype(">u2"), flags=8)) == ref
    assert bytes(c.encode(a.astype(">u2"))) == native     # by value
    # The bytes path, the dtype path and out= agree.
    raw = bytes(c.decode(ref, flags=12, bits_per_sample=16))
    assert raw[:a.nbytes] == a.tobytes()
    np.testing.assert_array_equal(
        c.decode(ref, flags=12, dtype="u2", shape=a.shape), a)
    np.testing.assert_array_equal(
        c.decode(ref, msb=True, dtype="u2", shape=a.shape), a)
    out = np.empty_like(a)
    c.decode(ref, flags=12, out=out)
    np.testing.assert_array_equal(out, a)
    np.testing.assert_array_equal(
        c.decode(ref, flags=8, dtype=">u2", shape=a.shape), a.astype(">u2"))
    # Read in the other byte order, 12-bit samples no longer fit;
    # imagecodecs drops their high bits, here it raises.
    twelve = np.array([0, 4095, 17, 2048] * 4, "u2")
    with pytest.raises(ValueError, match="AEC_DATA_MSB"):
        c.encode(twelve, bits_per_sample=12, flags=12)
    with pytest.raises(ValueError, match="AEC_DATA_MSB"):
        c.encode(twelve.astype(">u2"), bits_per_sample=12, flags=8)
    np.testing.assert_array_equal(
        c.decode(c.encode(twelve, bits_per_sample=12), bits_per_sample=12,
                 dtype="u2", shape=twelve.shape), twelve)


def test_aec_big_endian_signed_array_keeps_the_signed_flag():
    """A big-endian int16 array sets the signed flag as a native one
    does. imagecodecs leaves it off (and cannot read the result back
    into a big-endian int16 array); signed=False reads its stream."""
    _need("aec")
    c = oc.get_codec("aec")
    a = ARRAYS["aec_i2"]()
    be = a.astype(">i2")
    theirs = _blob(IC_BLOBS, "aec_i2_be_flags8")
    assert bytes(c.encode(be, flags=8, signed=False)) == theirs
    mine = bytes(c.encode(be, flags=8))
    assert mine != theirs
    np.testing.assert_array_equal(
        c.decode(mine, flags=8, dtype=">i2", shape=a.shape), be)
    np.testing.assert_array_equal(
        c.decode(theirs, flags=8, signed=False, dtype=">i2", shape=a.shape),
        be)


def test_aec_int8_is_coded_as_imagecodecs_codes_it():
    """imagecodecs codes int8 without the signed flag. Writing the same
    and reading int8 the same way makes the bytes equal and lets its
    int8 streams decode right with dtype='i1'."""
    _need("aec")
    a = ARRAYS["aec_i1"]()
    ref = _blob(IC_BLOBS, "aec_i1")
    c = oc.get_codec("aec")
    assert bytes(c.encode(a)) == ref
    np.testing.assert_array_equal(c.decode(ref, dtype="i1", shape=a.shape), a)
    out = np.empty_like(a)
    c.decode(ref, out=out)
    np.testing.assert_array_equal(out, a)
    # Signed coding is still there when asked for.
    signed = c.encode(a, signed=True)
    assert bytes(signed) != ref
    np.testing.assert_array_equal(
        c.decode(signed, signed=True, dtype="i1", shape=a.shape), a)
    # Negative samples narrower than 8 bits need it.
    with pytest.raises(ValueError, match="signed=True"):
        c.encode(np.array([-3, 2] * 8, "i1"), bits_per_sample=4)


def test_aec_pad_rsi_decodes_and_encodes_only_where_libaec_can():
    """A libaec built with ENABLE_RSI_PADDING pads each RSI with
    AEC_PAD_RSI (two bytes more here); one built without it accepts the
    flag but writes no padding, a stream no decoder reads back with the
    flag. Decoding the padded stream works either way; encoding with
    the flag raises where it cannot be honored."""
    _need("aec")
    a = ARRAYS["aec_u1_pad_rsi"]()
    ref = _blob(IC_BLOBS, "aec_u1_pad_rsi")
    c = oc.get_codec("aec")
    plain = bytes(c.encode(a, block_size=8, rsi=2))
    assert len(ref) == len(plain) + 2
    np.testing.assert_array_equal(
        c.decode(ref, pad_rsi=True, block_size=8, rsi=2, dtype="u1",
                 shape=a.shape), a)
    np.testing.assert_array_equal(
        c.decode(ref, flags=8 | 32, blocksize=8, rsi=2, dtype="u1",
                 shape=a.shape), a)
    try:
        padded = c.encode(a, pad_rsi=True, block_size=8, rsi=2)
    except ValueError as exc:
        assert "AEC_PAD_RSI" in str(exc)
        with pytest.raises(ValueError, match="AEC_PAD_RSI"):
            c.encode(a, flags=8 | 32, block_size=8, rsi=2)
    else:
        assert bytes(padded) == ref


@pytest.mark.parametrize("key,header,kw", [
    ("aec_u4_16_64", (256, 32, 16, 64, 8),
     {"block_size": 16, "rsi": 64, "bits_per_sample": 32}),
    ("aec_i4_nopre", (192, 32, 32, 128, 1), {"preprocess": False}),
    ("aec_u2", (192, 16, 32, 128, 8), {"block_size": 32, "rsi": 128}),
])
def test_aec_reads_legacy_blobs_given_their_parameters(key, header, kw):
    """0.4.0 ignored coding parameters on decode, so its callers may
    pass the ones a blob was written with. They agree with its
    preamble, so the blob still reads as one."""
    _need("aec")
    a = ARRAYS[key]()
    old = _blob(LEGACY_BLOBS, key)
    assert struct.unpack_from("<QBBHB3x", old) == header
    c = oc.get_codec("aec")
    assert bytes(c.decode(old, **kw)) == a.tobytes()
    np.testing.assert_array_equal(
        c.decode(old, dtype=a.dtype, shape=a.shape, **kw), a)
    out = np.empty_like(a)
    c.decode(old, out=out, **kw)
    np.testing.assert_array_equal(out, a)
    # imagecodecs' names for the same parameters
    names = {"block_size": "blocksize", "bits_per_sample": "bitspersample"}
    theirs = {names.get(k, k): v for k, v in kw.items()}
    np.testing.assert_array_equal(
        c.decode(old, dtype=a.dtype, shape=a.shape, **theirs), a)


# ---------------------------------------------------------------------------
# pcodec: standalone format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["pcodec_f4", "pcodec_i2"])
def test_pcodec_writes_and_reads_imagecodecs_bytes(key):
    _need("pcodec")
    a = ARRAYS[key]()
    ref = _blob(IC_BLOBS, key)
    c = oc.get_codec("pcodec")
    assert bytes(c.encode(a)) == ref
    flat = c.decode(ref)
    assert flat.dtype == a.dtype and flat.shape == (a.size,)
    np.testing.assert_array_equal(flat, a.ravel())
    np.testing.assert_array_equal(c.decode(ref, shape=a.shape), a)
    out = np.empty_like(a)
    assert c.decode(ref, out=out) is out
    np.testing.assert_array_equal(out, a)
    assert c.signature(ref)


def _pco_header(blob):
    """pcodec docs/format.md, standalone header: 'pco!', a version byte,
    the number type byte (version 2 on), then (version 3) the element
    count as 6 bits p and p + 1 bits of n, least significant bit first."""
    assert blob[:4] == b"pco!"
    version, number_type = blob[4], blob[5]
    bits = int.from_bytes(blob[6:16], "little")
    p = bits & 63
    return version, number_type, (bits >> 6) & ((1 << (p + 1)) - 1)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 100, 65536, 300000])
def test_pcodec_header_fields_by_hand(n):
    _need("pcodec")
    a = np.arange(n, dtype="f8")
    blob = bytes(oc.get_codec("pcodec").encode(a))
    version, number_type, count = _pco_header(blob)
    assert (version, number_type, count) == (3, 6, n)   # PCO_TYPE_F64 = 6
    back = oc.get_codec("pcodec").decode(blob)
    assert back.dtype == np.float64 and back.shape == (n,)
    np.testing.assert_array_equal(back, a)


def test_pcodec_checks_what_the_caller_says():
    _need("pcodec")
    c = oc.get_codec("pcodec")
    ref = _blob(IC_BLOBS, "pcodec_f4")
    with pytest.raises(ValueError):
        c.decode(ref, dtype="f8")
    with pytest.raises(ValueError):
        c.decode(ref, shape=(5, 5))
    with pytest.raises(ValueError):
        c.decode(b"junk" + ref)
    with pytest.raises(TypeError):
        c.encode(ARRAYS["pcodec_f4"](), quality=3)
    a = ARRAYS["pcodec_f4"]()
    assert c.encode(a, pagesize=8) == c.encode(a, max_page_n=8)
    assert c.encode(a, level=8) == ref


def test_pcodec_reads_legacy_blobs():
    _need("pcodec")
    a = ARRAYS["pcodec_f4"]()
    old = _blob(LEGACY_BLOBS, "pcodec_f4")
    assert old[:4] == b"PCOO" and old[72:] == _blob(IC_BLOBS, "pcodec_f4")
    c = oc.get_codec("pcodec")
    assert c.signature(old)
    back = c.decode(old)
    assert back.shape == a.shape and back.dtype == a.dtype
    np.testing.assert_array_equal(back, a)


def _pco_with_hint(blob, hint):
    """Rewrite the version-3 count hint, keeping its bit width p."""
    bits = int.from_bytes(blob[6:16], "little")
    p = bits & 63
    assert hint < 1 << (p + 1)
    mask = ((1 << (p + 1)) - 1) << 6
    bits = (bits & ~mask) | (hint << 6)
    return blob[:6] + bits.to_bytes(10, "little") + blob[16:]


@pytest.mark.parametrize("hint", [0, 23, 25, 31])
def test_pcodec_count_is_only_a_hint(hint):
    """pcodec docs/format.md: the standalone header holds the count of
    numbers "if known; 0 otherwise", and the data ends with a
    termination byte. shape or out sets the count; the hint is only
    the default."""
    _need("pcodec")
    c = oc.get_codec("pcodec")
    a = ARRAYS["pcodec_f4"]()                       # 24 values
    blob = _pco_with_hint(_blob(IC_BLOBS, "pcodec_f4"), hint)
    assert _pco_header(blob) == (3, 5, hint)
    np.testing.assert_array_equal(c.decode(blob, shape=a.shape), a)
    out = np.empty_like(a)
    assert c.decode(blob, out=out) is out
    np.testing.assert_array_equal(out, a)
    if hint == 0:
        with pytest.raises(ValueError, match="shape"):
            c.decode(blob)
    elif hint > a.size:
        np.testing.assert_array_equal(c.decode(blob), a.ravel())
    else:
        with pytest.raises(RuntimeError, match="shape="):
            c.decode(blob)
    with pytest.raises(ValueError, match="holds 24"):
        c.decode(blob, shape=(25,))
    with pytest.raises(RuntimeError):
        c.decode(blob, shape=(23,))


def test_pcodec_and_others_code_big_endian_arrays_by_value():
    """rcomp, pcodec and sperr code a big-endian array's values, the
    same stream as for the native array; imagecodecs codes its bytes as
    native numbers, and its stream of such an array decodes here to the
    byte-swapped numbers it holds."""
    for name, a, kw in (
            ("rcomp", ARRAYS["rcomp_i2"](), {}),
            ("pcodec", ARRAYS["pcodec_f4"](), {}),
            ("sperr", ARRAYS["sperr_f4_2d"](), {"mode": "pwe", "pwe": 1e-3})):
        if not oc.has_codec(name):
            continue
        c = oc.get_codec(name)
        be = a.astype(a.dtype.newbyteorder(">"))
        assert bytes(c.encode(be, **kw)) == bytes(c.encode(a, **kw))
    if oc.has_codec("rcomp"):
        a = ARRAYS["rcomp_i2"]()
        theirs = _blob(IC_BLOBS, "rcomp_be_i2")
        assert theirs != _blob(IC_BLOBS, "rcomp_i2")
        back = oc.get_codec("rcomp").decode(theirs, shape=a.shape, dtype=">i2")
        np.testing.assert_array_equal(back, a.byteswap())


# ---------------------------------------------------------------------------
# sz3: SZ3's own stream
# ---------------------------------------------------------------------------


def _sz3_layout(blob):
    """SZ3/api/sz.hpp: magic 0xF342F310, data version, payload size
    (uint64), payload, then Config::save: size byte, N, bit width, the
    dimensions packed at that width (LSB first, size-1 ones dropped),
    then num as uint64."""
    magic, version, size = struct.unpack_from("<IIQ", blob)
    conf = blob[16 + size:]
    ndim, width = conf[1], conf[2]
    nbytes = (ndim * width + 7) // 8
    packed = int.from_bytes(conf[3:3 + nbytes], "little")
    dims = tuple((packed >> (i * width)) & ((1 << width) - 1) for i in range(ndim))
    num = struct.unpack_from("<Q", conf, 3 + nbytes)[0]
    return magic, version, dims, num


@pytest.mark.parametrize("key", ["sz3_f4", "sz3_f8"])
def test_sz3_writes_and_reads_imagecodecs_bytes(key):
    _need("sz3")
    a = ARRAYS[key]()
    ref = _blob(IC_BLOBS, key)
    c = oc.get_codec("sz3")
    assert bytes(c.encode(a, mode="abs", abs_err=1e-3)) == ref
    assert bytes(c.encode(a, mode="abs", abs=1e-3)) == ref
    assert bytes(c.encode(a, mode=0, abs=1e-3)) == ref       # SZ3.MODE.ABS
    magic, _, dims, num = _sz3_layout(ref)
    assert magic == 0xF342F310 and dims == a.shape and num == a.size
    back = c.decode(ref, dtype=a.dtype)
    assert back.shape == a.shape and back.dtype == a.dtype
    assert _sha(back) == IC_DECODED_SHA256[key]
    assert c.signature(ref)


def test_sz3_needs_dtype_and_checks_shape():
    _need("sz3")
    c = oc.get_codec("sz3")
    ref = _blob(IC_BLOBS, "sz3_f4")
    with pytest.raises(ValueError, match="dtype"):
        c.decode(ref)
    # SZ3 drops dimensions of size 1; a shape that only adds them fits.
    assert c.decode(ref, dtype="f4", shape=(6, 1, 10)).shape == (6, 1, 10)
    with pytest.raises(ValueError):
        c.decode(ref, dtype="f4", shape=(10, 6))
    out = np.empty((6, 10), "f4")
    assert c.decode(ref, out=out) is out
    assert _sha(out) == IC_DECODED_SHA256["sz3_f4"]


def test_sz3_bad_streams_raise_instead_of_ending_the_process():
    """SZ3 throws (killing the process) on a stream it cannot read;
    the stream is checked first."""
    _need("sz3")
    c = oc.get_codec("sz3")
    ref = _blob(IC_BLOBS, "sz3_f4")
    for bad in (ref[:40], ref[:-5],
                ref[:4] + struct.pack("<I", 0x02000000) + ref[8:]):
        with pytest.raises(RuntimeError):
            c.decode(bad, dtype="f4")


def test_sz3_unsupported_requests_raise_in_process():
    """The SZ3 C API prints "not support" and exits the process for the
    PSNR and L2-norm modes and for integer data. Run in a child so a
    regression shows as a failure here rather than a silent exit."""
    _need("sz3")
    code = textwrap.dedent("""
        import numpy as np, opencodecs as oc
        c = oc.get_codec("sz3")
        a = np.zeros((4, 4), "f4")
        for call in (lambda: c.encode(a, mode="psnr"),
                     lambda: c.encode(a, mode="norm"),
                     lambda: c.encode(a, mode=4),
                     lambda: c.encode(a, psnr=60),
                     lambda: c.encode(np.zeros(9, "i4")),
                     lambda: c.encode(np.zeros((2, 2, 2, 2, 2), "f4"))):
            try:
                call()
            except (ValueError, TypeError):
                continue
            raise SystemExit("not refused")
        print("REFUSED")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0 and "REFUSED" in r.stdout, r.stdout + r.stderr


def test_sz3_reads_legacy_blobs():
    _need("sz3")
    old = _blob(LEGACY_BLOBS, "sz3_f4")
    assert old[:4] == b"SZ3O" and old[48:] == _blob(IC_BLOBS, "sz3_f4")
    back = oc.get_codec("sz3").decode(old)
    assert back.shape == (6, 10) and back.dtype == np.float32
    assert _sha(back) == IC_DECODED_SHA256["sz3_f4"]


@pytest.mark.parametrize("key", ["sz3_f4_unpred", "sz3_f8_unpred"])
def test_sz3_reads_streams_with_unpredictable_values(key):
    """Values SZ3 cannot predict are stored as they are, in the data
    type, so these streams show their type in their layout."""
    _need("sz3")
    a = ARRAYS[key]()
    ref = _blob(IC_BLOBS, key)
    c = oc.get_codec("sz3")
    assert bytes(c.encode(a, mode="abs", abs=1e-3)) == ref
    back = c.decode(ref, dtype=a.dtype)
    assert _sha(back) == IC_DECODED_SHA256[key]


@pytest.mark.parametrize("key,eb", [
    ("regression_f4", 1e-3), ("composed_f8", 1e-3),
    ("nopred_f8", 1e-7), ("regression_only_f4", 1e-7),
])
def test_sz3_reads_cpp_api_layouts(key, eb):
    _need("sz3")
    dt = key[-2:]
    ref = _blob(SZ3_CPP_BLOBS, key)
    a = _cpp_data(dt)
    back = oc.get_codec("sz3").decode(ref, dtype=dt)
    assert back.shape == a.shape and back.dtype == a.dtype
    assert float(np.abs(back.astype("f8") - a.astype("f8")).max()) <= eb


# Run in a child: before the payload was checked, each of these ended
# the process (SIGABRT, SIGSEGV) or read and wrote past SZ3's buffers.
_SZ3_REFUSALS = textwrap.dedent("""
    import pickle, struct, sys
    import numpy as np
    import opencodecs as oc
    blobs = pickle.load(open(sys.argv[1], "rb"))
    c = oc.get_codec("sz3")
    results = {}

    def outcome(call):
        try:
            call()
        except ValueError as exc:
            return "ValueError: " + str(exc)
        except RuntimeError as exc:
            return "RuntimeError: " + str(exc)
        return "decoded"

    for key, blob in blobs.items():
        if key.startswith("legacy"):
            continue
        other = "f8" if key.endswith("f4") or "_f4_" in key else "f4"
        results[key] = outcome(lambda: c.decode(blob, dtype=other))
    # Damage that the checks catch before SZ3 reads anything.
    blob = blobs["sz3_f8_unpred"]
    longer = blob[:16] + struct.pack("<Q", struct.unpack_from("<Q", blob, 16)[0] + 8) + blob[24:]
    results["inflated size"] = outcome(lambda: c.decode(longer, dtype="f8"))
    broken = bytearray(blob)
    broken[30:40] = bytes(10)
    results["damaged zstd frame"] = outcome(lambda: c.decode(bytes(broken), dtype="f8"))
    old = blobs["legacy"]
    bad_old = old[:4] + bytes([1 - old[4]]) + old[5:]   # SZ_FLOAT <-> SZ_DOUBLE
    results["legacy wrong type"] = outcome(lambda: c.decode(bad_old))
    print(repr(results))
""")


def test_sz3_wrong_dtype_and_damage_raise_instead_of_ending_the_process(tmp_path):
    _need("sz3")
    blobs = {k: _blob(IC_BLOBS, k) for k in ("sz3_f4", "sz3_f8_unpred", "sz3_f4_unpred")}
    blobs.update({k: _blob(SZ3_CPP_BLOBS, k) for k in SZ3_CPP_BLOBS})
    # A legacy blob whose preamble names the wrong type: f4 data, stored
    # with the preamble byte for float64.
    blobs["legacy"] = _blob(LEGACY_BLOBS, "sz3_f4")
    src = tmp_path / "blobs.pkl"
    with open(src, "wb") as fh:
        pickle.dump(blobs, fh)
    r = subprocess.run([sys.executable, "-c", _SZ3_REFUSALS, str(src)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.returncode, r.stderr[-2000:])
    import ast
    results = ast.literal_eval(r.stdout.strip().splitlines()[-1])
    for key in ("sz3_f8_unpred", "sz3_f4_unpred", "regression_f4",
                "composed_f8", "nopred_f8", "regression_only_f4", "sz3_f4"):
        assert results[key].startswith("ValueError: sz3 decode: the stream holds"), (
            key, results[key])
    for key in ("inflated size", "damaged zstd frame", "legacy wrong type"):
        assert results[key].startswith("RuntimeError"), (key, results[key])


# ---------------------------------------------------------------------------
# sperr: SPERR's own format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["sperr_f4_2d", "sperr_f8_3d"])
def test_sperr_writes_and_reads_imagecodecs_bytes(key):
    _need("sperr")
    a = ARRAYS[key]()
    ref = _blob(IC_BLOBS, key)
    c = oc.get_codec("sperr")
    assert bytes(c.encode(a, mode="pwe", pwe=1e-3)) == ref
    assert bytes(c.encode(a, mode="pwe", level=1e-3)) == ref
    assert bytes(c.encode(a, mode=3, level=1e-3)) == ref     # SPERR.MODE.PWE
    back = c.decode(ref)
    assert back.shape == a.shape and back.dtype == a.dtype
    assert _sha(back) == IC_DECODED_SHA256[key]


def test_sperr_header_fields_by_hand():
    """SPERR_C_API.h / SPERR3D_Stream_Tools: byte 0 the major version,
    byte 1 eight flags packed most significant first (0x40 3-D,
    0x20 float), then uint32 dimensions fastest first; a single-chunk
    3-D stream follows with one uint32 chunk length (14 + 4 bytes of
    header)."""
    _need("sperr")
    two = _blob(IC_BLOBS, "sperr_f4_2d")
    assert struct.unpack_from("<BII", two, 1) == (0x20, 16, 12)
    three = _blob(IC_BLOBS, "sperr_f8_3d")
    flags, x, y, z, chunk_len = struct.unpack_from("<BIIII", three, 1)
    assert (flags, x, y, z) == (0x40, 8, 6, 4)
    assert 18 + chunk_len == len(three)


def test_sperr_headerless_2d():
    _need("sperr")
    a = ARRAYS["sperr_f4_2d"]()
    ref = _blob(IC_BLOBS, "sperr_f4_2d")
    c = oc.get_codec("sperr")
    bare = bytes(c.encode(a, mode="pwe", pwe=1e-3, header=False))
    assert bare == ref[10:]         # the documented 10-byte header, stripped
    back = c.decode(bare, header=False, shape=a.shape, dtype=a.dtype)
    assert _sha(back) == IC_DECODED_SHA256["sperr_f4_2d"]
    with pytest.raises(ValueError):
        c.decode(bare, header=False)
    with pytest.raises(ValueError):
        c.encode(ARRAYS["sperr_f8_3d"](), header=False)


def test_sperr_checks_the_header_before_decoding():
    _need("sperr")
    c = oc.get_codec("sperr")
    three = _blob(IC_BLOBS, "sperr_f8_3d")
    for bad in (three[:12], three[:-3], three + b"\0",
                bytes([three[0] + 1]) + three[1:]):
        with pytest.raises(RuntimeError):
            c.decode(bad)
    with pytest.raises(ValueError):
        c.decode(three, shape=(4, 6, 9))
    as_f4 = c.decode(three, dtype="f4")
    assert as_f4.dtype == np.float32 and as_f4.shape == (4, 6, 8)


@pytest.mark.parametrize("key,shape,value", [
    ("sperr_const_2d", (5, 7), 2.5), ("sperr_const_3d", (3, 4, 5), -1.25)])
def test_sperr_reads_constant_fields(key, shape, value):
    """A constant field's coded data is just SPERR's 17-byte
    conditioner header: flags 0x81, the value count, the value."""
    _need("sperr")
    ref = _blob(IC_BLOBS, key)
    start = 10 if len(shape) == 2 else 18
    assert len(ref) == start + 17 and ref[start] == 0x81
    assert struct.unpack_from("<Qd", ref, start + 1) == (np.prod(shape), value)
    back = oc.get_codec("sperr").decode(ref)
    assert back.shape == shape
    np.testing.assert_array_equal(back, np.full(shape, value))


_SPERR_DAMAGE = textwrap.dedent("""
    import pickle, sys
    import opencodecs as oc
    from opencodecs.codecs import _sperr
    cases = pickle.load(open(sys.argv[1], "rb"))
    c = oc.get_codec("sperr")
    for name, blob, kw in cases:
        try:
            c.decode(blob, **kw)
        except _sperr.SperrError:
            continue
        raise SystemExit(f"decoded damaged stream {name}")
    print("ALL RAISED", len(cases))
""")


def test_sperr_damaged_fixed_fields_raise_instead_of_ending_the_process(tmp_path):
    """SPERR's decoder trusts every length and count in its input, and
    each of these ended the process (segmentation fault or abort, or an
    allocation of up to 2**64 bits). They are checked first now. The
    decode runs in a child so a regression cannot take pytest down."""
    _need("sperr")
    c = oc.get_codec("sperr")
    y, x = np.mgrid[0:64, 0:96]
    two = bytes(c.encode((np.sin(x / 7.0) + np.cos(y / 5.0)).astype("f4"),
                         mode="pwe", pwe=1e-3))
    three = _blob(IC_BLOBS, "sperr_f8_3d")
    const = _blob(IC_BLOBS, "sperr_const_2d")

    def flip(blob, pos, xor):
        return blob[:pos] + bytes([blob[pos] ^ xor]) + blob[pos + 1:]

    # 2-D: 10-byte header, 17-byte conditioner, 9-byte SPECK header.
    cases = [(f"2d[:{n}]", two[:n], {}) for n in range(11, 36)]
    cases += [
        ("2d conditioner flags", flip(two, 10, 0xFF), {}),
        ("2d bit planes", flip(two, 27, 0x80), {}),
        ("2d bit count", flip(two, 35, 0x40), {}),
        ("2d headerless[:20]", two[10:30], {"header": False, "shape": (64, 96),
                                             "dtype": "f4"}),
        ("3d flag cleared", flip(three, 1, 0x40), {}),
        ("3d bit count", flip(three, 18 + 17 + 8, 0x10), {}),
        ("constant + byte", const + b"\0", {}),
        ("constant count", flip(const, 11, 0x01), {}),
    ]
    src = tmp_path / "cases.pkl"
    with open(src, "wb") as fh:
        pickle.dump(cases, fh)
    r = subprocess.run([sys.executable, "-c", _SPERR_DAMAGE, str(src)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "ALL RAISED" in r.stdout, (
        r.returncode, r.stdout[-500:], r.stderr[-2000:])


def test_sperr_checks_pass_every_stream_sperr_writes():
    _need("sperr")
    c = oc.get_codec("sperr")
    rng = np.random.default_rng(5)
    arrays = [rng.random((33, 47)), rng.standard_cauchy((6, 7, 9)),
              np.sin(np.arange(16 * 16 * 16) / 7.0).reshape(16, 16, 16),
              np.zeros((8, 9)), np.full((2, 3, 4), 7.0)]
    for a in arrays:
        for dt in ("f4", "f8"):
            for kw in ({"mode": "pwe", "pwe": 1e-12}, {"mode": "pwe", "pwe": 1e-3},
                       {"mode": "psnr", "psnr": 300}, {"mode": "bpp", "bpp": 64},
                       {"mode": "bpp", "bpp": 0.5}):
                blob = c.encode(a.astype(dt), **kw)
                assert c.decode(blob).shape == a.shape
                if a.ndim == 3:
                    split = c.encode(a.astype(dt), chunks=(4, 4, 4), **kw)
                    assert c.decode(split).shape == a.shape
                else:
                    bare = c.encode(a.astype(dt), header=False, **kw)
                    assert c.decode(bare, header=False, shape=a.shape,
                                    dtype=dt).shape == a.shape


@pytest.mark.parametrize("key", ["sperr_f4_2d", "sperr_f8_3d"])
def test_sperr_reads_legacy_blobs(key):
    _need("sperr")
    old = _blob(LEGACY_BLOBS, key)
    assert old[:4] == b"SPRR"
    c = oc.get_codec("sperr")
    assert c.signature(old)
    back = c.decode(old)
    assert back.shape == ARRAYS[key]().shape
    assert _sha(back) == IC_DECODED_SHA256[key]


def test_sperr_compresses_a_volume_as_one_chunk_by_default():
    """imagecodecs passes the volume's own size as the chunk size, so
    SPERR writes one chunk: flags without 0x10 (several chunks) and a
    single uint32 chunk length after the 14-byte header. sperr3d's
    256 is there on request."""
    _need("sperr")
    c = oc.get_codec("sperr")
    a = np.sin(np.linspace(0, 30, 6 * 10 * 600)).reshape(6, 10, 600).astype("f4")
    one = bytes(c.encode(a, mode="pwe", pwe=1e-3))
    assert one == bytes(c.encode(a, mode="pwe", pwe=1e-3, chunks=a.shape))
    flags, x, y, z, chunk_len = struct.unpack_from("<BIIII", one, 1)
    assert (flags, x, y, z) == (0x40 | 0x20, 600, 10, 6)
    assert 18 + chunk_len == len(one)
    split = bytes(c.encode(a, mode="pwe", pwe=1e-3, chunks=(256, 256, 256)))
    assert split[1] & 0x10
    for blob in (one, split):
        back = c.decode(blob)
        assert float(np.abs(back - a).max()) <= 1.01e-3
    # An axis longer than 65535 fits one chunk; the 16-bit limit
    # applies only to the chunk sizes of a split volume.
    long = np.sin(np.linspace(0, 99, 70000 * 2 * 2)).reshape(70000, 2, 2)
    blob = bytes(c.encode(long, mode="pwe", pwe=1e-3))
    assert not blob[1] & 0x10
    assert c.decode(blob).shape == long.shape
    with pytest.raises(ValueError):
        c.encode(long, chunks=(70000, 1, 1))


# ---------------------------------------------------------------------------
# aec: signed samples narrower than their items, bytes input, 0-d arrays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key,bps,kw", [
    ("aec_i2_12", 12, {}),
    ("aec_i4_20", 20, {}),
    ("aec_i1_4", 4, {"signed": True}),
])
def test_aec_masks_signed_samples_to_bits_per_sample(key, bps, kw):
    """libaec takes a signed sample as a bits_per_sample-bit two's
    complement number (encode.c, preprocess_signed). imagecodecs wrote
    the references from the masked arrays, and decodes them to the
    arrays themselves; a signed array's sign-extended negatives code
    the same stream and read back as they were."""
    _need("aec")
    c = oc.get_codec("aec")
    a = ARRAYS[key]()
    ref = _blob(IC_BLOBS, key)
    assert bytes(c.encode(a, bits_per_sample=bps, **kw)) == ref
    back = c.decode(ref, bits_per_sample=bps, dtype=a.dtype, shape=a.shape,
                    **kw)
    np.testing.assert_array_equal(back, a)
    # The same samples big-endian, coded by value.
    be = a.astype(a.dtype.newbyteorder(">"))
    assert bytes(c.encode(be, bits_per_sample=bps, **kw)) == ref
    # Bytes are libaec's own samples: masked ones are taken as they
    # are, sign-extended ones are masked.
    mask = (1 << bps) - 1
    masked = (a.astype("i8") & mask).astype(a.dtype)
    flags = 8 | 1
    assert bytes(c.encode(masked.tobytes(), bits_per_sample=bps,
                          flags=flags)) == ref
    assert bytes(c.encode(a.tobytes(), bits_per_sample=bps,
                          flags=flags)) == ref
    # In an array, a value above the signed range raises rather than
    # being read back as a negative number.
    with pytest.raises(ValueError, match="signed=False"):
        c.encode(masked, bits_per_sample=bps, **kw)
    with pytest.raises(ValueError, match="do not fit"):
        c.encode(np.array([1 << (bps - 1), 0] * 8, a.dtype),
                 bits_per_sample=bps, **kw)
    with pytest.raises(ValueError, match="do not fit"):
        c.encode(np.array([-(1 << (bps - 1)) - 1, 0] * 8, a.dtype),
                 bits_per_sample=bps, **kw)


def test_aec_masks_three_byte_signed_samples():
    """AEC_DATA_3BYTE samples (bytes input only) are masked the same way."""
    _need("aec")
    c = oc.get_codec("aec")
    v = [-(1 << 19), -1, 0, 5, (1 << 19) - 1, -3] * 16
    flags = 8 | 2 | 1                      # preprocess, 3-byte, signed
    ext = b"".join((x & 0xFFFFFF).to_bytes(3, "little") for x in v)
    masked = b"".join((x & 0xFFFFF).to_bytes(3, "little") for x in v)
    stream = bytes(c.encode(ext, bits_per_sample=20, flags=flags))
    assert stream == bytes(c.encode(masked, bits_per_sample=20, flags=flags))
    d = c.decode(stream, bits_per_sample=20, flags=flags, out=len(ext))
    got = [int.from_bytes(d[i:i + 3], "little", signed=True)
           for i in range(0, len(d), 3)]
    assert got == v
    with pytest.raises(ValueError, match="do not fit"):
        c.encode((1 << 20).to_bytes(3, "little") * 8, bits_per_sample=20,
                 flags=8 | 2)


def test_aec_checks_bytes_input_against_bits_per_sample():
    """Bytes input above bits_per_sample raised nothing and wrote a
    stream that libaec then refused to decode."""
    _need("aec")
    c = oc.get_codec("aec")
    with pytest.raises(ValueError, match="do not fit"):
        c.encode(bytes(range(256)) * 4, bits_per_sample=4)
    data = bytes(i % 16 for i in range(256))
    blob = c.encode(data, bits_per_sample=4)
    assert c.decode(blob, bits_per_sample=4, out=len(data)) == data


@pytest.mark.parametrize("key", ["aec_0d_u1", "aec_0d_u2"])
def test_aec_codes_a_0d_array_as_one_sample(key):
    _need("aec")
    c = oc.get_codec("aec")
    a = ARRAYS[key]()
    ref = _blob(IC_BLOBS, key)
    assert bytes(c.encode(a)) == ref
    back = c.decode(ref, dtype=a.dtype, shape=())
    assert back.shape == () and back == a


# ---------------------------------------------------------------------------
# sz3: imagecodecs' defaults
# ---------------------------------------------------------------------------


def test_sz3_defaults_are_imagecodecs():
    """imagecodecs.sz3_encode(a) codes with mode ABS and an error bound
    of 0; a call with no bound writes the same bytes here."""
    _need("sz3")
    c = oc.get_codec("sz3")
    a = ARRAYS["sz3_f4_default"]()
    ref = _blob(IC_BLOBS, "sz3_f4_default")
    assert bytes(c.encode(a)) == ref
    assert bytes(c.encode(a, mode="abs", abs=0.0)) == ref
    np.testing.assert_array_equal(c.decode(ref, dtype=a.dtype), a)
    assert bytes(c.encode(a, abs_err=1e-3)) == _blob(IC_BLOBS, "sz3_f4")


def _sz3_hard_values(dt):
    fi = np.finfo(dt)
    rng = np.random.default_rng(8)
    special = np.array([np.nan, -0.0, 0.0, np.inf, -np.inf, fi.tiny,
                        fi.tiny / 4, -fi.tiny / 8, fi.max, -fi.max,
                        fi.eps, 1.0, -1.0], dt)
    lo, hi = np.log10(fi.tiny), np.log10(fi.max)
    spread = (10.0 ** rng.uniform(lo, hi, 200)
              * rng.choice([-1, 1], 200)).astype(dt)
    return {
        "special": np.resize(special, (9, 13)),
        "spread": spread.reshape(10, 20),
        "noise": rng.normal(size=(17, 23)).astype(dt),
        "constant": np.full((12, 11), 3.25, dt),
    }


def test_sz3_default_is_lossless(tmp_path):
    """The README marks sz3 lossless at its default (mode 'abs', bound
    0): every value, NaN, signed zero, subnormals and the extremes
    included, comes back with the same bits, and imagecodecs (the
    independent reference, in a child process) writes the same bytes
    and decodes them to the same bits."""
    _need("sz3")
    c = oc.get_codec("sz3")
    cases = {}
    for dt in ("f4", "f8"):
        for name, a in _sz3_hard_values(dt).items():
            blob = bytes(c.encode(a))
            back = c.decode(blob, dtype=a.dtype, shape=a.shape)
            assert back.tobytes() == a.tobytes(), (dt, name)
            cases[f"{dt}-{name}"] = (a, blob)
    if not _have_imagecodecs():
        pytest.skip("imagecodecs not installed")
    src, dst = tmp_path / "in.pkl", tmp_path / "out.pkl"
    with open(src, "wb") as fh:
        pickle.dump(cases, fh)
    child = textwrap.dedent("""
        import pickle, sys
        import imagecodecs as ic
        cases = pickle.load(open(sys.argv[1], "rb"))
        out = {k: (bytes(ic.sz3_encode(a)),
                   ic.sz3_decode(blob, a.shape, a.dtype).tobytes())
               for k, (a, blob) in cases.items()}
        pickle.dump(out, open(sys.argv[2], "wb"))
    """)
    r = subprocess.run([sys.executable, "-c", child, str(src), str(dst)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    with open(dst, "rb") as fh:
        theirs = pickle.load(fh)
    for key, (a, blob) in cases.items():
        their_blob, their_back = theirs[key]
        assert their_blob == blob, key
        assert their_back == a.tobytes(), key


# ---------------------------------------------------------------------------
# sperr: infinite quantization steps, recognition without format=
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["sperr_inf_q", "sperr_inf_q_3d"])
def test_sperr_refuses_to_write_an_infinite_step_and_reads_one(key):
    """In psnr mode SPERR's step overflows to infinity for values this
    large, and the stream decodes to NaN in imagecodecs and in 0.4.0.
    Writing one raises; reading one gives what those give."""
    _need("sperr")
    c = oc.get_codec("sperr")
    a = ARRAYS[key]()
    ref = _blob(IC_BLOBS, key)
    back = c.decode(ref)
    assert back.shape == a.shape and np.isnan(back).all()
    with pytest.raises(ValueError, match="infinity"):
        c.encode(a, mode="psnr", psnr=60)
    # pwe mode codes the same values.
    pwe = c.decode(c.encode(a, mode="pwe", pwe=1e190))
    assert float(np.abs(pwe - a).max()) <= 1.01e190
    if key in LEGACY_BLOBS:
        old = c.decode(_blob(LEGACY_BLOBS, key))
        assert old.shape == a.shape and np.isnan(old).all()


def test_sperr_streams_are_recognized_without_format():
    """oc.read finds SPERR's own streams by their header fields, as it
    found the 0.4.0 'SPRR' preamble, and leaves other streams alone."""
    _need("sperr")
    from opencodecs.core.codec import codec_for_bytes
    c = oc.get_codec("sperr")
    rng = np.random.default_rng(4)
    blobs = [_blob(IC_BLOBS, k) for k in
             ("sperr_f4_2d", "sperr_f8_3d", "sperr_const_2d", "sperr_const_3d",
              "sperr_inf_q", "sperr_inf_q_3d")]
    big2 = np.sin(np.linspace(0, 40, 90 * 70)).reshape(90, 70) + rng.random((90, 70))
    vol = np.sin(np.linspace(0, 40, 6 * 30 * 40)).reshape(6, 30, 40).astype("f4")
    blobs += [bytes(c.encode(big2, mode="pwe", pwe=1e-4)),
              bytes(c.encode(vol, mode="pwe", pwe=1e-4)),
              bytes(c.encode(vol, mode="pwe", pwe=1e-4, chunks=(4, 8, 8))),
              # bpp streams are cut short of the bit counts they declare
              bytes(c.encode(big2, mode="bpp", bpp=6)),
              bytes(c.encode(vol, mode="bpp", bpp=6))]
    assert all(len(b) > 512 for b in blobs[-5:])
    for blob in blobs:
        assert codec_for_bytes(blob).name == "sperr"
        got, want = oc.read(blob), c.decode(blob)
        assert got.shape == want.shape
        np.testing.assert_array_equal(got, want)
    # A headerless 2-D stream and the other codecs' streams are not
    # claimed.
    others = [bytes(c.encode(big2, mode="pwe", pwe=1e-4, header=False))]
    others += [_blob(IC_BLOBS, k) for k in IC_BLOBS
               if not k.startswith("sperr")]
    for blob in others:
        assert not c.signature(blob[:512]), blob[:16].hex()
    # Random bytes that start like a SPERR header are not taken for one.
    for n in (40, 300, 512):
        for _ in range(2000):
            b = bytearray(rng.integers(0, 256, n, dtype=np.uint8).tobytes())
            b[0], b[1] = 0, int(rng.choice([0x00, 0x20, 0x40, 0x60, 0x70]))
            assert not c.signature(bytes(b)), bytes(b[:24]).hex()
    # Damage to the fields checked turns recognition off.
    two = _blob(IC_BLOBS, "sperr_f4_2d")
    for i, v in ((0, 1), (1, 0x21), (10, 0x02)):
        bad = bytearray(two)
        bad[i] = v
        assert not c.signature(bytes(bad))


# Volumes whose chunk table (4 bytes a chunk after a 20-byte header)
# pushes the first coded stream's fields past the 512 bytes oc.read
# sniffs: 117 chunks and more, up to tables longer than the window.
_LONG_TABLES = [((8, 64, 64), (2, 8, 8), {}),          # 256 chunks
                ((7, 61, 59), (2, 8, 8), {}),          # 168, merged remainders
                ((117, 4, 4), (1, 4, 4), {}),          # 117
                ((118, 4, 4), (1, 4, 4), {}),          # 118
                ((122, 4, 4), (1, 4, 4), {}),          # 122, table ends at 508
                ((123, 4, 4), (1, 4, 4), {}),          # 123, table fills it
                ((300, 4, 4), (1, 4, 4), {"const": True})]  # constant chunks


def _long_table_volume(shape, kw, seed):
    if kw.get("const"):
        return np.full(shape, 2.5, "f4")
    return np.random.default_rng(seed).normal(size=shape).astype("f4")


def test_sperr_long_chunk_tables_are_recognized(tmp_path):
    """A volume of many chunks is recognized without format=, as its
    0.4.0 'SPRR' blob was by magic, for streams written by opencodecs
    and by imagecodecs (sperr_encode with chunks=), which oc.read must
    decode to imagecodecs' values."""
    _need("sperr")
    from opencodecs.core.codec import codec_for_bytes
    c = oc.get_codec("sperr")
    ours = []
    for i, (shape, chunks, kw) in enumerate(_LONG_TABLES):
        a = _long_table_volume(shape, kw, i)
        for mode, level in (("pwe", 0.5), ("bpp", 0.05), ("psnr", 40.0)):
            blob = bytes(c.encode(a, mode=mode, level=level, chunks=chunks))
            assert codec_for_bytes(blob).name == "sperr", (shape, mode)
            np.testing.assert_array_equal(oc.read(blob), c.decode(blob))
            ours.append(blob)
    # Damaging one chunk length the window holds turns recognition off.
    blob = ours[0]
    for at, value in ((20, 0), (24, 20), (400, 0xFFFFFFFF)):
        bad = bytearray(blob[:512])
        struct.pack_into("<I", bad, at, value)
        assert not c.signature(bytes(bad)), (at, value)
    # So do random chunk lengths after a plausible header.
    rng = np.random.default_rng(5)
    for _ in range(2000):
        bad = bytearray(blob[:20]) + rng.integers(0, 256, 492, np.uint8).tobytes()
        assert not c.signature(bytes(bad))
    if not _have_imagecodecs():
        pytest.skip("imagecodecs not installed")
    cases = [(_long_table_volume(shape, kw, i), chunks)
             for i, (shape, chunks, kw) in enumerate(_LONG_TABLES)]
    src, dst = tmp_path / "in.pkl", tmp_path / "out.pkl"
    with open(src, "wb") as fh:
        pickle.dump(cases, fh)
    child = textwrap.dedent("""
        import pickle, sys
        import imagecodecs as ic
        cases = pickle.load(open(sys.argv[1], "rb"))
        out = []
        for a, chunks in cases:
            blob = bytes(ic.sperr_encode(a, 0.5, "pwe", chunks=chunks))
            out.append((blob, ic.sperr_decode(blob)))
        pickle.dump(out, open(sys.argv[2], "wb"))
    """)
    r = subprocess.run([sys.executable, "-c", child, str(src), str(dst)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    with open(dst, "rb") as fh:
        theirs = pickle.load(fh)
    for blob, want in theirs:
        assert codec_for_bytes(blob).name == "sperr"
        np.testing.assert_array_equal(oc.read(blob), want)


# ---------------------------------------------------------------------------
# A live exchange with imagecodecs, in a child process, both directions.
# ---------------------------------------------------------------------------


def _have_imagecodecs():
    return importlib.util.find_spec("imagecodecs") is not None


def _child_imagecodecs_version():
    """imagecodecs' version as a child process sees it, or None."""
    r = subprocess.run(
        [sys.executable, "-c", "import imagecodecs; print(imagecodecs.__version__)"],
        capture_output=True, text=True, timeout=120)
    if r.returncode:
        return None
    try:
        return tuple(int(x) for x in r.stdout.strip().split(".")[:3])
    except ValueError:
        return None


_IC_SIDE = textwrap.dedent("""
    import pickle, sys
    import numpy as np
    import imagecodecs as ic
    cases, ours = pickle.load(open(sys.argv[1], "rb"))
    decoded, theirs = {}, {}
    for key, (codec, a, kw) in cases.items():
        if key.endswith("-be"):
            # Pickling turns a big-endian array native; turn it back.
            a = a.astype(a.dtype.newbyteorder(">"))
        if codec == "rcomp":
            theirs[key] = bytes(ic.rcomp_encode(a, **kw))
            decoded[key] = ic.rcomp_decode(ours[key], a.shape, a.dtype, **kw)
        elif codec == "aec":
            theirs[key] = bytes(ic.aec_encode(a, **kw))
            # imagecodecs encodes int8 unsigned but sets the signed
            # flag when decoding into int8; decode its bytes as uint8.
            out = np.empty(a.shape, "u1") if a.dtype == np.int8 else np.empty_like(a)
            decoded[key] = ic.aec_decode(ours[key], out=out, **kw).view(a.dtype)
        elif codec == "pcodec":
            theirs[key] = bytes(ic.pcodec_encode(a))
            decoded[key] = ic.pcodec_decode(ours[key], a.shape, a.dtype)
        elif codec == "sz3":
            theirs[key] = bytes(ic.sz3_encode(a, mode="abs", abs=1e-3))
            decoded[key] = ic.sz3_decode(ours[key], a.shape, a.dtype)
        elif codec == "sperr":
            theirs[key] = bytes(ic.sperr_encode(a, 1e-3, "pwe"))
            decoded[key] = ic.sperr_decode(ours[key])
    pickle.dump((decoded, theirs), open(sys.argv[2], "wb"))
""")


def _exchange_cases():
    rng = np.random.default_rng(11)
    cases = {}
    for dt in ("i1", "u1", "i2", "u2", "i4", "u4"):
        info = np.iinfo(dt)
        a = rng.integers(max(info.min, -900), min(info.max, 900), (31, 17)).astype(dt)
        cases[f"rcomp-{dt}"] = ("rcomp", a, {})
        cases[f"rcomp-{dt}-n16"] = ("rcomp", a, {"nblock": 16})
    for dt in ("i1", "u1", "i2", "u2", "i4", "u4"):
        # A whole number of 32-sample blocks: imagecodecs.aec_decode
        # refuses padding past an exact out= buffer, even its own.
        b = np.cumsum(rng.integers(-3, 4, 4096)).astype(dt)
        cases[f"aec-{dt}"] = ("aec", b, {})
        cases[f"aec-{dt}-32"] = ("aec", b, {"blocksize": 32, "rsi": 128})
        # An explicit flags keeps its byte order bit (4, AEC_DATA_MSB):
        # the bytes are coded in that order on both sides.
        cases[f"aec-{dt}-msb"] = ("aec", b, {"flags": 8 | 4})
        # imagecodecs leaves the signed flag off for a big-endian int16
        # or int32 array (and then misreads its own stream); opencodecs
        # sets it for either byte order, so compare unsigned ones.
        if b.itemsize > 1 and dt[0] == "u":
            cases[f"aec-{dt}-flags-be"] = (
                "aec", b.astype(b.dtype.newbyteorder(">")), {"flags": 8})
    for dt in ("u1", "i1", "u2", "i2", "f2", "u4", "i4", "f4", "u8", "i8", "f8"):
        cases[f"pcodec-{dt}"] = ("pcodec", (rng.random((7, 9, 5)) * 100).astype(dt), {})
    for shape in ((1000,), (33, 47), (5, 6, 7), (2, 3, 4, 5)):
        for dt in ("f4", "f8"):
            a = np.sin(np.arange(np.prod(shape)) / 7.0).reshape(shape).astype(dt)
            cases[f"sz3-{dt}-{len(shape)}d"] = ("sz3", a, {})
    for dt in ("f4", "f8"):
        cases[f"sperr-{dt}-2d"] = (
            "sperr", np.sin(np.linspace(0, 9, 70 * 50)).reshape(70, 50).astype(dt), {})
        cases[f"sperr-{dt}-3d"] = (
            "sperr", np.sin(np.linspace(0, 9, 9 * 20 * 30)).reshape(9, 20, 30).astype(dt), {})
    # Longer than 256 along x: one chunk by default on both sides.
    # sperr3d's chunking splits it; those bytes need only decode.
    wide = np.sin(np.linspace(0, 30, 12 * 20 * 600)).reshape(12, 20, 600).astype("f4")
    cases["sperr-f4-wide"] = ("sperr", wide, {})
    cases["sperr-f4-wide-chunked"] = ("sperr", wide, {"chunks": (256, 256, 256)})
    return cases


def _oc_encode(codec, a, kw):
    c = oc.get_codec(codec)
    if codec == "sz3":
        return bytes(c.encode(a, mode="abs", abs=1e-3))
    if codec == "sperr":
        return bytes(c.encode(a, mode="pwe", level=1e-3, **kw))
    return bytes(c.encode(a, **kw))


def _oc_decode(codec, blob, a, kw):
    c = oc.get_codec(codec)
    if codec == "rcomp":
        return c.decode(blob, shape=a.shape, dtype=a.dtype, **kw)
    if codec == "aec":
        return c.decode(blob, dtype=a.dtype, shape=a.shape, **kw)
    if codec == "pcodec":
        return c.decode(blob, shape=a.shape)
    if codec == "sz3":
        return c.decode(blob, dtype=a.dtype, shape=a.shape)
    return c.decode(blob)


@pytest.mark.skipif(not _have_imagecodecs(), reason="imagecodecs not installed")
def test_live_exchange_with_imagecodecs(tmp_path):
    """Our bytes equal imagecodecs' bytes, and each library decodes the
    other's. imagecodecs runs in a child process so the two libraries'
    statically linked copies of shared C code never meet in one
    process."""
    version = _child_imagecodecs_version()
    if version is None or version < (2026, 8, 16):
        # Older releases link older SZ3 and pcodec builds, whose stream
        # versions differ; the fixtures above pin 2026.8.16.
        pytest.skip(f"needs imagecodecs >= 2026.8.16, found {version}")
    cases = {k: v for k, v in _exchange_cases().items() if oc.has_codec(v[0])}
    ours = {k: _oc_encode(*v) for k, v in cases.items()}
    src, dst = tmp_path / "in.pkl", tmp_path / "out.pkl"
    with open(src, "wb") as fh:
        pickle.dump((cases, ours), fh)
    r = subprocess.run([sys.executable, "-c", _IC_SIDE, str(src), str(dst)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    with open(dst, "rb") as fh:
        decoded, theirs = pickle.load(fh)
    problems = []
    for key, (codec, a, kw) in cases.items():
        if ours[key] != theirs[key] and not key.endswith("-chunked"):
            problems.append(f"{key}: bytes differ")
        mine = np.asarray(_oc_decode(codec, theirs[key], a, kw))
        other = np.asarray(decoded[key])
        if codec in ("rcomp", "aec", "pcodec"):
            if not (np.array_equal(mine, a) and mine.dtype == a.dtype):
                problems.append(f"{key}: opencodecs misread imagecodecs")
            if not np.array_equal(other.reshape(a.shape), a):
                problems.append(f"{key}: imagecodecs misread opencodecs")
        elif key.endswith("-chunked"):
            for got in (mine, other):
                if float(np.abs(got.astype("f8") - a.astype("f8")).max()) > 1.01e-3:
                    problems.append(f"{key}: error bound exceeded")
        else:
            if mine.shape != a.shape or not np.array_equal(mine, other):
                problems.append(f"{key}: decodes of one stream differ")
            if float(np.abs(mine.astype("f8") - a.astype("f8")).max()) > 1.01e-3:
                problems.append(f"{key}: error bound exceeded")
    assert not problems, problems


_IC_PCO_HINTS = textwrap.dedent("""
    import pickle, sys
    import numpy as np
    import imagecodecs as ic
    a = np.cos(np.arange(1000) / 9.0).astype("f4")
    blob = bytes(ic.pcodec_encode(a))
    bits = int.from_bytes(blob[6:16], "little")
    p = bits & 63
    mask = ((1 << (p + 1)) - 1) << 6
    result = {}
    for hint in (0, 999, 1001, 1000):
        b = bits & ~mask | hint << 6
        s = blob[:6] + b.to_bytes(10, "little") + blob[16:]
        result[hint] = (s, ic.pcodec_decode(s, shape=(1000,), dtype="f4"))
    pickle.dump((a, result), open(sys.argv[1], "wb"))
""")


@pytest.mark.skipif(not _have_imagecodecs(), reason="imagecodecs not installed")
def test_pcodec_count_hint_against_imagecodecs(tmp_path):
    """imagecodecs reads a stream whose count hint is 0 (unknown, as
    the format allows) or wrong, given the shape; so does opencodecs."""
    _need("pcodec")
    version = _child_imagecodecs_version()
    if version is None or version < (2026, 8, 16):
        pytest.skip(f"needs imagecodecs >= 2026.8.16, found {version}")
    dst = tmp_path / "pco.pkl"
    r = subprocess.run([sys.executable, "-c", _IC_PCO_HINTS, str(dst)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    with open(dst, "rb") as fh:
        a, result = pickle.load(fh)
    c = oc.get_codec("pcodec")
    for hint, (blob, theirs) in result.items():
        np.testing.assert_array_equal(theirs, a)
        np.testing.assert_array_equal(c.decode(blob, shape=(1000,)), a)
        out = np.empty(1000, "f4")
        np.testing.assert_array_equal(c.decode(blob, out=out), a)
        if hint >= 1000:
            np.testing.assert_array_equal(c.decode(blob), a)
