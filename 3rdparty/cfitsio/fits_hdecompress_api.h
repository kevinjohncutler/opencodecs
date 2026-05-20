/* Minimal external header for opencodecs's Cython binding. The two
 * functions below are defined in 3rdparty/cfitsio/fits_hdecompress.c
 * but cfitsio doesn't ship a clean header for them — we only need
 * decode entry points so list them directly. */
#ifndef OPENCODECS_FITS_HDECOMPRESS_API_H
#define OPENCODECS_FITS_HDECOMPRESS_API_H

#include "fitsio2.h"   /* LONGLONG typedef + ffpmsg / FFLOCK stubs */

#ifdef __cplusplus
extern "C" {
#endif

int fits_hdecompress(
    unsigned char *input, int smooth, int *a, int na,
    int *ny, int *nx, int *scale, int *status);

int fits_hdecompress64(
    unsigned char *input, int smooth, LONGLONG *a, int na,
    int *ny, int *nx, int *scale, int *status);

#ifdef __cplusplus
}
#endif

#endif
