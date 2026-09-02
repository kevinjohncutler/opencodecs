/* Declarations for the Rice compressor/decompressor in ricecomp.c.
 *
 * cfitsio ships these functions without a public header of their own,
 * so this file exists to give the opencodecs Cython binding something
 * to declare against. It is written for opencodecs; the implementation
 * in ricecomp.c is cfitsio's (see License.txt).
 *
 * Return convention: the functions return a byte count on success, or
 * one of the negative codes below. cfitsio's originals call ffpmsg and
 * return a status through a global; the negative codes let the binding
 * raise a specific Python exception without linking the rest of
 * cfitsio, and ricecomp.c is adapted to use them.
 */

#ifndef OPENCODECS_RICECOMP_H
#define OPENCODECS_RICECOMP_H

#define RCOMP_OK            (0)
#define RCOMP_ERROR_MEMORY  (-1)   /* allocation failed */
#define RCOMP_ERROR_EOB     (-2)   /* ran off the end of the output buffer */
#define RCOMP_ERROR_EOS     (-3)   /* input stream ended mid-symbol */
#define RCOMP_WARN_UNUSED   (-4)   /* input left over after decoding */

/* Compress nx samples from a[] into at most clen bytes of c[], in
 * blocks of nblock samples. Returns the compressed length. */
int rcomp_int(int a[], int nx, unsigned char *c, int clen, int nblock);
int rcomp_short(short a[], int nx, unsigned char *c, int clen, int nblock);
int rcomp_byte(signed char a[], int nx, unsigned char *c, int clen, int nblock);

/* Decompress clen bytes of c[] into nx samples of array[], in blocks of
 * nblock samples. Returns 0 on success. */
int rdecomp_int(unsigned char *c, int clen, unsigned int array[],
                int nx, int nblock);
int rdecomp_short(unsigned char *c, int clen, unsigned short array[],
                  int nx, int nblock);
int rdecomp_byte(unsigned char *c, int clen, unsigned char array[],
                 int nx, int nblock);

#endif  /* OPENCODECS_RICECOMP_H */
