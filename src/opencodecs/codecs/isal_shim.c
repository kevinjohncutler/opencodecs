/* opencodecs/_isal_shim.c
 *
 * Minimal C shim that wraps Intel's ISA-L (igzip) deflate/inflate
 * behind a flat one-shot byte-in / byte-out API. Hides the heavy
 * isal_zstream / inflate_state structs from Cython so the pyx
 * binding is a tight wrapper around two function calls.
 *
 * Wire format is the same zlib stream (RFC 1950: 2-byte zlib header +
 * deflate payload + adler32 footer) that ``zlib.compress`` and
 * ``libdeflate_zlib_compress`` produce. ISA-L's ``ISAL_ZLIB`` flag
 * handles header/footer emission on the encode side and verification
 * on the decode side.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

#include <isa-l/igzip_lib.h>

/* Encode ``src`` (length ``srcsize``) into ``dst`` (capacity ``dstcap``)
 * as a zlib-format stream. Returns the number of bytes written or
 * a negative ISAL_* error code (see igzip_lib.h). ``level`` is one
 * of 0/1/2/3 — ISA-L only defines four compression levels, the input
 * is clamped to [0, 3] here. */
ssize_t
opencodecs_isal_zlib_encode(
    const uint8_t* src, size_t srcsize,
    uint8_t* dst, size_t dstcap,
    int level)
{
    struct isal_zstream stream;
    isal_deflate_stateless_init(&stream);

    /* ISA-L levels 0..3; level_buf size per level is documented in
     * igzip_lib.h. Higher levels need bigger workspace buffers. */
    if (level < 0) level = 0;
    if (level > 3) level = 3;
    const uint32_t level_buf_sizes[4] = {
        ISAL_DEF_LVL0_DEFAULT,
        ISAL_DEF_LVL1_DEFAULT,
        ISAL_DEF_LVL2_DEFAULT,
        ISAL_DEF_LVL3_DEFAULT,
    };
    uint32_t level_buf_size = level_buf_sizes[level];
    uint8_t* level_buf = (uint8_t*)malloc(level_buf_size);
    if (!level_buf) return -100;

    stream.level = (uint32_t)level;
    stream.level_buf = level_buf;
    stream.level_buf_size = level_buf_size;
    stream.gzip_flag = IGZIP_ZLIB;   /* emit zlib header + adler32 footer */
    stream.flush = NO_FLUSH;
    stream.end_of_stream = 1;        /* one-shot encode */
    stream.next_in = (uint8_t*)src;
    stream.avail_in = (uint32_t)srcsize;
    stream.next_out = dst;
    stream.avail_out = (uint32_t)dstcap;

    int rc = isal_deflate_stateless(&stream);
    free(level_buf);
    if (rc != COMP_OK) return (ssize_t)rc;
    return (ssize_t)stream.total_out;
}

/* Decode a zlib-format stream from ``src`` into ``dst``. Returns the
 * number of bytes written or a negative error code. ``dstcap`` must
 * be at least the uncompressed size; caller learns the expected size
 * from out-of-band metadata or by retrying with a bigger buffer.
 *
 * Uses the streaming ``isal_inflate`` API in a loop — the simpler
 * ``isal_inflate_stateless`` only handles a single deflate block,
 * which works for incompressible input (one literal block) but
 * silently truncates after the first block on real compressible
 * input. */
ssize_t
opencodecs_isal_zlib_decode(
    const uint8_t* src, size_t srcsize,
    uint8_t* dst, size_t dstcap)
{
    struct inflate_state state;
    isal_inflate_init(&state);
    state.crc_flag = ISAL_ZLIB;
    state.next_in = (uint8_t*)src;
    state.avail_in = (uint32_t)srcsize;
    state.next_out = dst;
    state.avail_out = (uint32_t)dstcap;

    /* Drive the decoder until it hits ISAL_BLOCK_FINISH (stream end)
     * or runs out of input / output. ``isal_inflate`` returns
     * ISAL_DECOMP_OK after every successful chunk; the
     * ``block_state`` field tells us when we're truly done. */
    while (1) {
        int rc = isal_inflate(&state);
        if (rc != ISAL_DECOMP_OK) return (ssize_t)rc;
        if (state.block_state == ISAL_BLOCK_FINISH) break;
        if (state.avail_in == 0 && state.avail_out > 0) {
            /* Input exhausted before the stream finished — caller's
             * buffer was complete but the input is truncated. */
            return -3;
        }
        if (state.avail_out == 0) {
            /* Output buffer full but stream not done — caller needs
             * a bigger buffer. Map to ISAL_OUT_OVERFLOW (-2) which
             * the pyx retry loop already understands. */
            return -2;
        }
    }
    return (ssize_t)state.total_out;
}
