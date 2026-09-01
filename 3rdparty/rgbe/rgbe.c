/* THIS CODE CARRIES NO GUARANTEE OF USABILITY OR FITNESS FOR ANY PURPOSE.
 * WHILE THE AUTHORS HAVE TRIED TO ENSURE THE PROGRAM WORKS CORRECTLY,
 * IT IS STRICTLY USE AT YOUR OWN RISK.  */

#include "rgbe.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>

/* This file contains code to read and write four byte rgbe file format
 developed by Greg Ward.  It handles the conversions between rgbe and
 pixels consisting of floats.  The data is assumed to be an array of floats.
 By default there are three floats per pixel in the order red, green, blue.
 (RGBE_DATA_??? values control this.)  Only the minimal header reading and
 writing is implemented.  Each routine does error checking and will return
 a status value as defined below.  This code is intended as a skeleton so
 feel free to modify it to suit your needs.

 (Place notice here if you modified the code.)
 posted to http://www.graphics.cornell.edu/~bjw/
 written by Bruce Walter  (bjw@graphics.cornell.edu)  5/26/95
 based on code written by Greg Ward

Modifications by Denis Mentey (denis@goortom.com):
Little modifications (type casts) to compile without warnings have been added.
Magic token check uncommented.
Header parsing errors fixed.

Modifications by Christoph Gohlke:

- Lint code.
- Replace FILE handling with stream.
- Simplify error handling.
- Fix buffer overflow and partial line handling in rgbe_stream_gets.
- Add const qualifiers to write functions.
- Use standard __cplusplus macro.

http://www.graphics.cornell.edu/online/formats/rgbe/

*/

#ifdef __cplusplus
/* define if your compiler understands inline commands */
#define INLINE inline
#else
#define INLINE
#endif

/* offsets to red, green, and blue components in a data (float) pixel */
#define RGBE_DATA_RED 0
#define RGBE_DATA_GREEN 1
#define RGBE_DATA_BLUE 2
/* number of floats per pixel */
#define RGBE_DATA_SIZE 3
/* maximum line length in header */
#define RGBE_HEADER_LINE_LENGTH 1024

/* standard conversion from float pixels to rgbe pixels */
/* note: you can remove the "inline"s if your compiler complains about it */
/* opencodecs: the encoder's counterpart to oc_rgbe_exp_scale.
   float2rgbe computed frexp(v, &e) * 256.0f / v per pixel, another
   libm call in a hot loop. frexp's mantissa divided by v is just
   2**-e, so the whole expression is 2**(8-e), and for a positive
   normal float e is the biased exponent minus 126. That makes the
   scale a table lookup on the exponent byte: 2**(134 - biased).

   Entries below index 7 would overflow a float, and are 0.0f
   because they are unreachable: the caller rejects v < 1e-32,
   which is about 2**-106, so the biased exponent is at least 21. */
static const float oc_rgbe_enc_scale[256] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0x1p127f, 0x1p126f, 0x1p125f, 0x1p124f, 0x1p123f,
    0x1p122f, 0x1p121f, 0x1p120f, 0x1p119f, 0x1p118f, 0x1p117f,
    0x1p116f, 0x1p115f, 0x1p114f, 0x1p113f, 0x1p112f, 0x1p111f,
    0x1p110f, 0x1p109f, 0x1p108f, 0x1p107f, 0x1p106f, 0x1p105f,
    0x1p104f, 0x1p103f, 0x1p102f, 0x1p101f, 0x1p100f, 0x1p99f,
    0x1p98f, 0x1p97f, 0x1p96f, 0x1p95f, 0x1p94f, 0x1p93f,
    0x1p92f, 0x1p91f, 0x1p90f, 0x1p89f, 0x1p88f, 0x1p87f,
    0x1p86f, 0x1p85f, 0x1p84f, 0x1p83f, 0x1p82f, 0x1p81f,
    0x1p80f, 0x1p79f, 0x1p78f, 0x1p77f, 0x1p76f, 0x1p75f,
    0x1p74f, 0x1p73f, 0x1p72f, 0x1p71f, 0x1p70f, 0x1p69f,
    0x1p68f, 0x1p67f, 0x1p66f, 0x1p65f, 0x1p64f, 0x1p63f,
    0x1p62f, 0x1p61f, 0x1p60f, 0x1p59f, 0x1p58f, 0x1p57f,
    0x1p56f, 0x1p55f, 0x1p54f, 0x1p53f, 0x1p52f, 0x1p51f,
    0x1p50f, 0x1p49f, 0x1p48f, 0x1p47f, 0x1p46f, 0x1p45f,
    0x1p44f, 0x1p43f, 0x1p42f, 0x1p41f, 0x1p40f, 0x1p39f,
    0x1p38f, 0x1p37f, 0x1p36f, 0x1p35f, 0x1p34f, 0x1p33f,
    0x1p32f, 0x1p31f, 0x1p30f, 0x1p29f, 0x1p28f, 0x1p27f,
    0x1p26f, 0x1p25f, 0x1p24f, 0x1p23f, 0x1p22f, 0x1p21f,
    0x1p20f, 0x1p19f, 0x1p18f, 0x1p17f, 0x1p16f, 0x1p15f,
    0x1p14f, 0x1p13f, 0x1p12f, 0x1p11f, 0x1p10f, 0x1p9f,
    0x1p8f, 0x1p7f, 0x1p6f, 0x1p5f, 0x1p4f, 0x1p3f,
    0x1p2f, 0x1p1f, 0x1p0f, 0x1p-1f, 0x1p-2f, 0x1p-3f,
    0x1p-4f, 0x1p-5f, 0x1p-6f, 0x1p-7f, 0x1p-8f, 0x1p-9f,
    0x1p-10f, 0x1p-11f, 0x1p-12f, 0x1p-13f, 0x1p-14f, 0x1p-15f,
    0x1p-16f, 0x1p-17f, 0x1p-18f, 0x1p-19f, 0x1p-20f, 0x1p-21f,
    0x1p-22f, 0x1p-23f, 0x1p-24f, 0x1p-25f, 0x1p-26f, 0x1p-27f,
    0x1p-28f, 0x1p-29f, 0x1p-30f, 0x1p-31f, 0x1p-32f, 0x1p-33f,
    0x1p-34f, 0x1p-35f, 0x1p-36f, 0x1p-37f, 0x1p-38f, 0x1p-39f,
    0x1p-40f, 0x1p-41f, 0x1p-42f, 0x1p-43f, 0x1p-44f, 0x1p-45f,
    0x1p-46f, 0x1p-47f, 0x1p-48f, 0x1p-49f, 0x1p-50f, 0x1p-51f,
    0x1p-52f, 0x1p-53f, 0x1p-54f, 0x1p-55f, 0x1p-56f, 0x1p-57f,
    0x1p-58f, 0x1p-59f, 0x1p-60f, 0x1p-61f, 0x1p-62f, 0x1p-63f,
    0x1p-64f, 0x1p-65f, 0x1p-66f, 0x1p-67f, 0x1p-68f, 0x1p-69f,
    0x1p-70f, 0x1p-71f, 0x1p-72f, 0x1p-73f, 0x1p-74f, 0x1p-75f,
    0x1p-76f, 0x1p-77f, 0x1p-78f, 0x1p-79f, 0x1p-80f, 0x1p-81f,
    0x1p-82f, 0x1p-83f, 0x1p-84f, 0x1p-85f, 0x1p-86f, 0x1p-87f,
    0x1p-88f, 0x1p-89f, 0x1p-90f, 0x1p-91f, 0x1p-92f, 0x1p-93f,
    0x1p-94f, 0x1p-95f, 0x1p-96f, 0x1p-97f, 0x1p-98f, 0x1p-99f,
    0x1p-100f, 0x1p-101f, 0x1p-102f, 0x1p-103f, 0x1p-104f, 0x1p-105f,
    0x1p-106f, 0x1p-107f, 0x1p-108f, 0x1p-109f, 0x1p-110f, 0x1p-111f,
    0x1p-112f, 0x1p-113f, 0x1p-114f, 0x1p-115f, 0x1p-116f, 0x1p-117f,
    0x1p-118f, 0x1p-119f, 0x1p-120f, 0x1p-121f,
};

static INLINE void
float2rgbe(
    unsigned char rgbe[4],
    float red,
    float green,
    float blue)
{
    float v;

    v = red;
    if (green > v)
        v = green;
    if (blue > v)
        v = blue;
    if (v < 1e-32) {
        rgbe[0] = rgbe[1] = rgbe[2] = rgbe[3] = 0;
    }
    else {
        unsigned int bits;
        unsigned int biased;
        memcpy(&bits, &v, sizeof(bits));
        biased = (bits >> 23) & 0xFFu;
        v = oc_rgbe_enc_scale[biased];
        rgbe[0] = (unsigned char)(red * v);
        rgbe[1] = (unsigned char)(green * v);
        rgbe[2] = (unsigned char)(blue * v);
        /* e + 128 where e == biased - 126 */
        rgbe[3] = (unsigned char)(biased + 2u);
    }
}

/* standard conversion from rgbe to float pixels */
/* note: Ward uses ldexp(col+0.5,exp-(128+8)).  However we wanted pixels */
/*       in the range [0,1] to map back into the range [0,1].            */
/* opencodecs: the original called ldexp() once per pixel, a libm call
   in the hot loop of every decode. All 256 scale factors are
   2**(e-136), so they fit in a compile-time table. OpenImageIO's
   HDR reader does the same thing; a static const table beats
   building one lazily because it needs no init call, no ready flag
   and no thread-safety argument.

   Entry 0 is 0.0f rather than 2**-136, which folds away the
   nonzero-pixel branch: a zero exponent must produce a zero pixel,
   and multiplying by 0.0f does that without a test.

   Hex float literals so the values are exact; the low entries are
   denormal and a decimal round-trip is easy to get subtly wrong. */
static const float oc_rgbe_exp_scale[256] = {
    0.0f, 0x1p-135f, 0x1p-134f, 0x1p-133f, 0x1p-132f, 0x1p-131f,
    0x1p-130f, 0x1p-129f, 0x1p-128f, 0x1p-127f, 0x1p-126f, 0x1p-125f,
    0x1p-124f, 0x1p-123f, 0x1p-122f, 0x1p-121f, 0x1p-120f, 0x1p-119f,
    0x1p-118f, 0x1p-117f, 0x1p-116f, 0x1p-115f, 0x1p-114f, 0x1p-113f,
    0x1p-112f, 0x1p-111f, 0x1p-110f, 0x1p-109f, 0x1p-108f, 0x1p-107f,
    0x1p-106f, 0x1p-105f, 0x1p-104f, 0x1p-103f, 0x1p-102f, 0x1p-101f,
    0x1p-100f, 0x1p-99f, 0x1p-98f, 0x1p-97f, 0x1p-96f, 0x1p-95f,
    0x1p-94f, 0x1p-93f, 0x1p-92f, 0x1p-91f, 0x1p-90f, 0x1p-89f,
    0x1p-88f, 0x1p-87f, 0x1p-86f, 0x1p-85f, 0x1p-84f, 0x1p-83f,
    0x1p-82f, 0x1p-81f, 0x1p-80f, 0x1p-79f, 0x1p-78f, 0x1p-77f,
    0x1p-76f, 0x1p-75f, 0x1p-74f, 0x1p-73f, 0x1p-72f, 0x1p-71f,
    0x1p-70f, 0x1p-69f, 0x1p-68f, 0x1p-67f, 0x1p-66f, 0x1p-65f,
    0x1p-64f, 0x1p-63f, 0x1p-62f, 0x1p-61f, 0x1p-60f, 0x1p-59f,
    0x1p-58f, 0x1p-57f, 0x1p-56f, 0x1p-55f, 0x1p-54f, 0x1p-53f,
    0x1p-52f, 0x1p-51f, 0x1p-50f, 0x1p-49f, 0x1p-48f, 0x1p-47f,
    0x1p-46f, 0x1p-45f, 0x1p-44f, 0x1p-43f, 0x1p-42f, 0x1p-41f,
    0x1p-40f, 0x1p-39f, 0x1p-38f, 0x1p-37f, 0x1p-36f, 0x1p-35f,
    0x1p-34f, 0x1p-33f, 0x1p-32f, 0x1p-31f, 0x1p-30f, 0x1p-29f,
    0x1p-28f, 0x1p-27f, 0x1p-26f, 0x1p-25f, 0x1p-24f, 0x1p-23f,
    0x1p-22f, 0x1p-21f, 0x1p-20f, 0x1p-19f, 0x1p-18f, 0x1p-17f,
    0x1p-16f, 0x1p-15f, 0x1p-14f, 0x1p-13f, 0x1p-12f, 0x1p-11f,
    0x1p-10f, 0x1p-9f, 0x1p-8f, 0x1p-7f, 0x1p-6f, 0x1p-5f,
    0x1p-4f, 0x1p-3f, 0x1p-2f, 0x1p-1f, 0x1p0f, 0x1p1f,
    0x1p2f, 0x1p3f, 0x1p4f, 0x1p5f, 0x1p6f, 0x1p7f,
    0x1p8f, 0x1p9f, 0x1p10f, 0x1p11f, 0x1p12f, 0x1p13f,
    0x1p14f, 0x1p15f, 0x1p16f, 0x1p17f, 0x1p18f, 0x1p19f,
    0x1p20f, 0x1p21f, 0x1p22f, 0x1p23f, 0x1p24f, 0x1p25f,
    0x1p26f, 0x1p27f, 0x1p28f, 0x1p29f, 0x1p30f, 0x1p31f,
    0x1p32f, 0x1p33f, 0x1p34f, 0x1p35f, 0x1p36f, 0x1p37f,
    0x1p38f, 0x1p39f, 0x1p40f, 0x1p41f, 0x1p42f, 0x1p43f,
    0x1p44f, 0x1p45f, 0x1p46f, 0x1p47f, 0x1p48f, 0x1p49f,
    0x1p50f, 0x1p51f, 0x1p52f, 0x1p53f, 0x1p54f, 0x1p55f,
    0x1p56f, 0x1p57f, 0x1p58f, 0x1p59f, 0x1p60f, 0x1p61f,
    0x1p62f, 0x1p63f, 0x1p64f, 0x1p65f, 0x1p66f, 0x1p67f,
    0x1p68f, 0x1p69f, 0x1p70f, 0x1p71f, 0x1p72f, 0x1p73f,
    0x1p74f, 0x1p75f, 0x1p76f, 0x1p77f, 0x1p78f, 0x1p79f,
    0x1p80f, 0x1p81f, 0x1p82f, 0x1p83f, 0x1p84f, 0x1p85f,
    0x1p86f, 0x1p87f, 0x1p88f, 0x1p89f, 0x1p90f, 0x1p91f,
    0x1p92f, 0x1p93f, 0x1p94f, 0x1p95f, 0x1p96f, 0x1p97f,
    0x1p98f, 0x1p99f, 0x1p100f, 0x1p101f, 0x1p102f, 0x1p103f,
    0x1p104f, 0x1p105f, 0x1p106f, 0x1p107f, 0x1p108f, 0x1p109f,
    0x1p110f, 0x1p111f, 0x1p112f, 0x1p113f, 0x1p114f, 0x1p115f,
    0x1p116f, 0x1p117f, 0x1p118f, 0x1p119f,
};

static INLINE void
rgbe2float(
    float *red,
    float *green,
    float *blue,
    unsigned char rgbe[4])
{
    const float f = oc_rgbe_exp_scale[rgbe[3]];
    *red = rgbe[0] * f;
    *green = rgbe[1] * f;
    *blue = rgbe[2] * f;
}

/* default minimal header. modify if you want more information in header */
int
RGBE_WriteHeader(
    rgbe_stream_t *fp,
    int width,
    int height,
    const rgbe_header_info *info)
{
    const char *programtype = "RADIANCE";

    if (info && (info->valid & RGBE_VALID_PROGRAMTYPE))
        programtype = info->programtype;
    if (rgbe_stream_printf(fp, "#?%s\n", programtype) < 0)
        return RGBE_WRITE_ERROR;
    /* The #? is to identify file type, the programtype is optional. */
    if (info && (info->valid & RGBE_VALID_GAMMA)) {
        if (rgbe_stream_printf(fp, "GAMMA=%g\n", info->gamma) < 0)
            return RGBE_WRITE_ERROR;
    }
    if (info && (info->valid & RGBE_VALID_EXPOSURE)) {
        if (rgbe_stream_printf(fp, "EXPOSURE=%g\n", info->exposure) < 0)
            return RGBE_WRITE_ERROR;
    }
    if (rgbe_stream_printf(fp, "FORMAT=32-bit_rle_rgbe\n\n") < 0)
        return RGBE_WRITE_ERROR;
    if (rgbe_stream_printf(fp, "-Y %d +X %d\n", height, width) < 0)
        return RGBE_WRITE_ERROR;
    return RGBE_RETURN_SUCCESS;
}

/* minimal header reading.  modify if you want to parse more information */
int
RGBE_ReadHeaderOriented(
    rgbe_stream_t *fp,
    int *width,
    int *height,
    rgbe_header_info *info,
    int *orientation)
{
    char buf[RGBE_HEADER_LINE_LENGTH];
    int found_format;
    float tempf;
    ssize_t i;

    found_format = 0;
    if (info) {
        info->valid = 0;
        info->programtype[0] = 0;
        info->gamma = info->exposure = 1.0;
    }
    if (rgbe_stream_gets(buf, sizeof(buf) / sizeof(buf[0]), fp) == NULL)
        return RGBE_READ_ERROR;
    if ((buf[0] != '#') || (buf[1] != '?')) {
        /* to require the magic token then uncomment the next line */
        /* return RGBE_FORMAT_ERROR; */ /* bad initial token */
    }
    else if (info) {
        info->valid |= RGBE_VALID_PROGRAMTYPE;
        for (i = 0; i < (ssize_t)sizeof(info->programtype) - 1; i++) {
            if ((buf[i + 2] == 0) || isspace(buf[i + 2]))
                break;
            info->programtype[i] = buf[i + 2];
        }
        info->programtype[i] = 0;
        if (rgbe_stream_gets(buf, sizeof(buf) / sizeof(buf[0]), fp) == 0)
            return RGBE_READ_ERROR;
    }
    for (;;) {
        if ((buf[0] == 0) || (buf[0] == '\n')) {
            if (found_format == 0) {
                return RGBE_FORMAT_ERROR; /* no FORMAT specifier found */
            }
            else {
                break;
            }
        }
        else if (
            (strcmp(buf, "FORMAT=32-bit_rle_rgbe\n") == 0) ||
            (strcmp(buf, "FORMAT=32-bit_rle_xyze\n") == 0))
        {
            found_format = 1;
        }
        else if (info && (sscanf(buf, "GAMMA=%g", &tempf) == 1)) {
            info->gamma = tempf;
            info->valid |= RGBE_VALID_GAMMA;
        }
        else if (info && (sscanf(buf, "EXPOSURE=%g", &tempf) == 1)) {
            info->exposure = tempf;
            info->valid |= RGBE_VALID_EXPOSURE;
        }
        if (rgbe_stream_gets(buf, sizeof(buf) / sizeof(buf[0]), fp) == 0)
            return RGBE_READ_ERROR;
    }
    if (strcmp(buf, "\n") != 0)
        /* missing blank line after FORMAT specifier */
        return RGBE_FORMAT_ERROR;
    if (rgbe_stream_gets(buf, sizeof(buf) / sizeof(buf[0]), fp) == 0)
        return RGBE_READ_ERROR;
    {
        /* Parse both axis tokens generically instead of matching only
           "-Y %d +X %d". The signs select the direction of each axis and
           the token order selects row- or column-major storage; see the
           RGBE_ORIENT_* notes in rgbe.h. */
        char sign1, axis1, sign2, axis2;
        int count1, count2;
        int orient = RGBE_ORIENT_NONE;

        if (sscanf(buf, " %c%c %d %c%c %d",
                   &sign1, &axis1, &count1,
                   &sign2, &axis2, &count2) != 6)
            return RGBE_FORMAT_ERROR; /* missing image size specifier */
        if ((sign1 != '+' && sign1 != '-') || (sign2 != '+' && sign2 != '-'))
            return RGBE_FORMAT_ERROR;
        if (axis1 == 'Y' && axis2 == 'X') {
            *height = count1;
            *width = count2;
            if (sign1 == '+') orient |= RGBE_ORIENT_FLIP_Y;
            if (sign2 == '-') orient |= RGBE_ORIENT_FLIP_X;
        }
        else if (axis1 == 'X' && axis2 == 'Y') {
            *width = count1;
            *height = count2;
            orient |= RGBE_ORIENT_TRANSPOSE;
            if (sign1 == '-') orient |= RGBE_ORIENT_FLIP_X;
            if (sign2 == '+') orient |= RGBE_ORIENT_FLIP_Y;
        }
        else {
            return RGBE_FORMAT_ERROR;
        }
        if (*width <= 0 || *height <= 0)
            return RGBE_FORMAT_ERROR;
        if (orientation != NULL)
            *orientation = orient;
        else if (orient != RGBE_ORIENT_NONE)
            return RGBE_FORMAT_ERROR;  /* caller wants the standard form */
    }
    return RGBE_RETURN_SUCCESS;
}

/* Standard-orientation reader, for callers that cannot handle a flipped
   or transposed image: anything but "-Y H +X W" is a format error. */
int
RGBE_ReadHeader(
    rgbe_stream_t *fp,
    int *width,
    int *height,
    rgbe_header_info *info)
{
    return RGBE_ReadHeaderOriented(fp, width, height, info, NULL);
}

/* simple write routine that does not use run-length encoding */
/* These routines can be made faster by allocating a larger buffer and
   fread-ing and fwrite-ing the data in larger chunks */
int
RGBE_WritePixels(
    rgbe_stream_t *fp,
    const float *data,
    int numpixels)
{
    unsigned char rgbe[4];

    while (numpixels-- > 0) {
        float2rgbe(
            rgbe,
            data[RGBE_DATA_RED],
            data[RGBE_DATA_GREEN],
            data[RGBE_DATA_BLUE]
        );
        data += RGBE_DATA_SIZE;
        if (rgbe_stream_write(rgbe, sizeof(rgbe), 1, fp) < 1)
            return RGBE_WRITE_ERROR;
    }
    return RGBE_RETURN_SUCCESS;
}

/* simple read routine.  will not correctly handle run-length encoding */
int
RGBE_ReadPixels(
    rgbe_stream_t *fp,
    float *data,
    int numpixels)
{
    unsigned char rgbe[4];

    while (numpixels-- > 0) {
        if (rgbe_stream_read(rgbe, sizeof(rgbe), 1, fp) < 1)
            return RGBE_READ_ERROR;
        rgbe2float(
            &data[RGBE_DATA_RED],
            &data[RGBE_DATA_GREEN],
            &data[RGBE_DATA_BLUE],
            rgbe
        );
        data += RGBE_DATA_SIZE;
    }
    return RGBE_RETURN_SUCCESS;
}

/* The code below is only needed for the run-length encoded files. */
/* Run length encoding adds considerable complexity but does */
/* save some space.  For each scanline, each channel (r,g,b,e) is */
/* encoded separately for better compression. */

static int
RGBE_WriteBytes_RLE(
    rgbe_stream_t *fp,
    unsigned char *data,
    int numbytes)
{
#define MINRUNLENGTH 4
    int cur, beg_run, run_count, old_run_count, nonrun_count;
    unsigned char buf[2];

    cur = 0;
    while (cur < numbytes) {
        beg_run = cur;
        /* find next run of length at least 4 if one exists */
        run_count = old_run_count = 0;
        while ((run_count < MINRUNLENGTH) && (beg_run < numbytes)) {
            beg_run += run_count;
            old_run_count = run_count;
            run_count = 1;
            while (
                (beg_run + run_count < numbytes) && (run_count < 127) &&
                (data[beg_run] == data[beg_run + run_count]))
                run_count++;
        }
        /* if data before next big run is a short run then write it as such */
        if ((old_run_count > 1) && (old_run_count == beg_run - cur)) {
            buf[0] = (unsigned char)(128 + old_run_count); /* short run */
            buf[1] = data[cur];
            if (rgbe_stream_write(buf, sizeof(buf[0]) * 2, 1, fp) < 1)
                return RGBE_WRITE_ERROR;
            cur = beg_run;
        }
        /* write out bytes until we reach the start of the next run */
        while (cur < beg_run) {
            nonrun_count = beg_run - cur;
            if (nonrun_count > 128)
                nonrun_count = 128;
            buf[0] = (unsigned char)nonrun_count;
            if (rgbe_stream_write(buf, sizeof(buf[0]), 1, fp) < 1)
                return RGBE_WRITE_ERROR;
            if (
                rgbe_stream_write(
                &data[cur],
                sizeof(data[0]) * nonrun_count,
                1,
                fp
                ) < 1)
                return RGBE_WRITE_ERROR;
            cur += nonrun_count;
        }
        /* write out next run if one was found */
        if (run_count >= MINRUNLENGTH) {
            buf[0] = (unsigned char)(128 + run_count);
            buf[1] = data[beg_run];
            if (rgbe_stream_write(buf, sizeof(buf[0]) * 2, 1, fp) < 1)
                return RGBE_WRITE_ERROR;
            cur += run_count;
        }
    }
    return RGBE_RETURN_SUCCESS;
#undef MINRUNLENGTH
}

int
RGBE_WritePixels_RLE(
    rgbe_stream_t *fp,
    const float *data,
    int scanline_width,
    int num_scanlines)
{
    unsigned char rgbe[4];
    unsigned char *buffer;
    int i, err;

    if ((scanline_width < 8) || (scanline_width > 0x7fff))
        /* run length encoding is not allowed so write flat */
        return RGBE_WritePixels(fp, data, scanline_width * num_scanlines);
    buffer =
        (unsigned char *)malloc(sizeof(unsigned char) * 4 * scanline_width);
    if (buffer == NULL)
        /* no buffer space so write flat */
        return RGBE_WritePixels(fp, data, scanline_width * num_scanlines);
    while (num_scanlines-- > 0) {
        rgbe[0] = 2;
        rgbe[1] = 2;
        rgbe[2] = (unsigned char)(scanline_width >> 8);
        rgbe[3] = (unsigned char)(scanline_width & 0xFF);
        if (rgbe_stream_write(rgbe, sizeof(rgbe), 1, fp) < 1) {
            free(buffer);
            return RGBE_WRITE_ERROR;
        }
        for (i = 0; i < scanline_width; i++) {
            float2rgbe(
                rgbe,
                data[RGBE_DATA_RED],
                data[RGBE_DATA_GREEN],
                data[RGBE_DATA_BLUE]
            );
            buffer[i] = rgbe[0];
            buffer[i + scanline_width] = rgbe[1];
            buffer[i + 2 * scanline_width] = rgbe[2];
            buffer[i + 3 * scanline_width] = rgbe[3];
            data += RGBE_DATA_SIZE;
        }
        /* write out each of the four channels separately run length encoded */
        /* first red, then green, then blue, then exponent */
        for (i = 0; i < 4; i++) {
            if (
                (err = RGBE_WriteBytes_RLE(
                    fp,
                    &buffer[i * scanline_width],
                    scanline_width
                )) !=
                RGBE_RETURN_SUCCESS)
            {
                free(buffer);
                return err;
            }
        }
    }
    free(buffer);
    return RGBE_RETURN_SUCCESS;
}

int
RGBE_ReadPixels_RLE(
    rgbe_stream_t *fp,
    float *data,
    int scanline_width,
    int num_scanlines)
{
    unsigned char rgbe[4], *scanline_buffer, *ptr, *ptr_end;
    int i, count;
    unsigned char buf[2];

    if ((scanline_width < 8) || (scanline_width > 0x7fff))
        /* run length encoding is not allowed so read flat */
        return RGBE_ReadPixels(fp, data, scanline_width * num_scanlines);
    scanline_buffer = NULL;
    /* read in each successive scanline */
    while (num_scanlines > 0) {
        if (rgbe_stream_read(rgbe, sizeof(rgbe), 1, fp) < 1) {
            free(scanline_buffer);
            return RGBE_READ_ERROR;
        }
        if ((rgbe[0] != 2) || (rgbe[1] != 2) || (rgbe[2] & 0x80)) {
            /* this file is not run length encoded */
            rgbe2float(&data[0], &data[1], &data[2], rgbe);
            data += RGBE_DATA_SIZE;
            free(scanline_buffer);
            return RGBE_ReadPixels(
                fp,
                data,
                scanline_width * num_scanlines - 1
            );
        }
        if ((((int)rgbe[2]) << 8 | rgbe[3]) != scanline_width) {
            free(scanline_buffer);
            return RGBE_FORMAT_ERROR; /* wrong scanline width */
        }
        if (scanline_buffer == NULL)
            scanline_buffer = (unsigned char *)malloc(
                sizeof(unsigned char) * 4 * scanline_width
            );
        if (scanline_buffer == NULL)
            return RGBE_MEMORY_ERROR; /* unable to allocate buffer space */

        ptr = &scanline_buffer[0];
        /* read each of the four channels for the scanline into the buffer */
        for (i = 0; i < 4; i++) {
            ptr_end = &scanline_buffer[(i + 1) * scanline_width];
            while (ptr < ptr_end) {
                if (rgbe_stream_read(buf, sizeof(buf[0]) * 2, 1, fp) < 1) {
                    free(scanline_buffer);
                    return RGBE_READ_ERROR;
                }
                if (buf[0] > 128) {
                    /* a run of the same value */
                    count = buf[0] - 128;
                    if ((count == 0) || (count > ptr_end - ptr)) {
                        free(scanline_buffer);
                        return RGBE_FORMAT_ERROR; /* bad scanline data */
                    }
                    while (count-- > 0) *ptr++ = buf[1];
                }
                else {
                    /* a non-run */
                    count = buf[0];
                    if ((count == 0) || (count > ptr_end - ptr)) {
                        free(scanline_buffer);
                        return RGBE_FORMAT_ERROR; /* bad scanline data */
                    }
                    *ptr++ = buf[1];
                    if (--count > 0) {
                        if (
                            rgbe_stream_read(
                            ptr,
                            sizeof(*ptr) * count,
                            1,
                            fp
                            ) < 1)
                        {
                            free(scanline_buffer);
                            return RGBE_READ_ERROR;
                        }
                        ptr += count;
                    }
                }
            }
        }
        /* now convert data from buffer into floats */
        for (i = 0; i < scanline_width; i++) {
            rgbe[0] = scanline_buffer[i];
            rgbe[1] = scanline_buffer[i + scanline_width];
            rgbe[2] = scanline_buffer[i + 2 * scanline_width];
            rgbe[3] = scanline_buffer[i + 3 * scanline_width];
            rgbe2float(
                &data[RGBE_DATA_RED],
                &data[RGBE_DATA_GREEN],
                &data[RGBE_DATA_BLUE],
                rgbe
            );
            data += RGBE_DATA_SIZE;
        }
        num_scanlines--;
    }
    free(scanline_buffer);
    return RGBE_RETURN_SUCCESS;
}

/* FILE-like memory-stream */

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

rgbe_stream_t *
rgbe_stream_new(
    size_t size,
    char *data)
{
    rgbe_stream_t *stream = (rgbe_stream_t *)malloc(sizeof(rgbe_stream_t));
    if (stream == NULL) {
        return NULL;
    }
    if (data == NULL) {
        stream->data = (char *)malloc(size);
        if (stream->data == NULL) {
            free(stream);
            return NULL;
        }
        stream->owner = 1;
    }
    else {
        stream->data = data;
        stream->owner = 0;
    }
    stream->size = size;
    stream->pos = 0;
    return stream;
}

void
rgbe_stream_del(
    rgbe_stream_t *stream)
{
    if (stream != NULL) {
        if ((stream->owner != 0) && (stream->data != NULL)) {
            free(stream->data);
        }
        free(stream);
    }
}

size_t
rgbe_stream_read(
    void *ptr,
    size_t size,
    size_t nmemb,
    rgbe_stream_t *stream)
{
    /*
    if ((stream == NULL) || (ptr == NULL)) {
        return 0;
    }
    */
    size *= nmemb;
    if (stream->pos + size > stream->size) {
        return 0;
    }
    memcpy(ptr, stream->data + stream->pos, size);
    stream->pos += size;
    return size;
}

size_t
rgbe_stream_write(
    const void *ptr,
    size_t size,
    size_t nmemb,
    rgbe_stream_t *stream)
{
    /*
    if ((stream == NULL) || (ptr == NULL)) {
        return 0;
    }
    */
    size *= nmemb;
    if (stream->pos + size > stream->size) {
        return 0;
    }
    memcpy(stream->data + stream->pos, ptr, size);
    stream->pos += size;
    return size;
}

int
rgbe_stream_printf(
    rgbe_stream_t *stream,
    const char *format,
    ...)
{
    va_list args;
    int size;

    /*
    if ((stream == NULL) || (format == NULL)) {
        return 0;
    }
    */
    va_start(args, format);
    size = vsnprintf(
        stream->data + stream->pos,
        stream->size - stream->pos,
        format,
        args
    );
    va_end(args);
    if ((size < 0) || (((size_t)size + 1) > stream->size - stream->pos)) {
        return -1;
    }
    stream->pos += size;
    return size;
}

char *
rgbe_stream_gets(
    char *str,
    size_t n,
    rgbe_stream_t *stream)
{
    size_t size;
    char *pch = NULL;
    char *data = NULL;

    /*
    if ((stream == NULL) || (str == NULL)) {
        return NULL;
    }
    */
    if ((n < 2) || (stream->size <= stream->pos)) {
        return NULL;
    }

    data = stream->data + stream->pos;
    size = MIN(n - 1, stream->size - stream->pos);
    pch = (char *)memchr((const void *)(data), '\n', size);
    if (pch != NULL) {
        size = pch - data + 1;
    }
    memcpy(str, data, size);
    str[size] = '\0';
    stream->pos += size;
    return str;
}
