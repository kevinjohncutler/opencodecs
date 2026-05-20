/* fitsio2.h — minimal cfitsio-internals stub for opencodecs.
 *
 * fits_hcompress.c / fits_hdecompress.c reference a small set of
 * cfitsio-internal symbols (LONGLONG, ffpmsg, FFLOCK/FFUNLOCK). Pulling
 * in the real cfitsio headers would drag the whole library in;
 * stubbing them here lets us vendor the two source files standalone
 * (BSD-style cfitsio license, see License.txt).
 */
#ifndef OPENCODECS_CFITSIO_STUB_H
#define OPENCODECS_CFITSIO_STUB_H

#include <stdio.h>

/* cfitsio uses LONGLONG as its portable 64-bit signed int. We use
 * ``long long`` (always >= 64-bit per C99) rather than ``int64_t``
 * because the latter is typedef'd to ``long`` on LP64 Linux but
 * ``long long`` on macOS — that platform-dependent typedef breaks
 * strict pointer compatibility with our Cython ``long long *``
 * declarations. */
typedef long long LONGLONG;

/* cfitsio's internal thread-safety macros guard reentrant globals.
 * The source uses them as bare statements (``FFLOCK;``), not
 * function-call form. We don't share state across opencodecs calls
 * (each decode allocates its own staging buffers), so noop both. */
#define FFLOCK   ((void)0)
#define FFUNLOCK ((void)0)

/* cfitsio's printf-like error logger. Route to stderr — keeps the
 * surface visible during decode bugs without taking a hard dep on
 * libcfitsio. */
static inline void
ffpmsg(const char *msg)
{
    if (msg) {
        fprintf(stderr, "opencodecs hcompress: %s\n", msg);
    }
}

/* Error codes that fits_hdecompress.c returns. cfitsio defines these
 * in fitsio.h with a sprawling enum of all libcfitsio errors; we only
 * need the one the H-decompress path emits. */
#ifndef DATA_DECOMPRESSION_ERR
#define DATA_DECOMPRESSION_ERR 414
#endif
#ifndef MEMORY_ALLOCATION
#define MEMORY_ALLOCATION 113
#endif

#endif  /* OPENCODECS_CFITSIO_STUB_H */
