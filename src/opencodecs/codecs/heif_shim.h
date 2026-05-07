/* heif_shim.h
 *
 * Adds C typedefs for libheif's struct types so Cython can declare variables
 * as `heif_error err;` instead of `struct heif_error err;`. Mac's libheif
 * header already provides these; Ubuntu's headers don't.
 *
 * Modern C lets us declare `typedef struct X X;` even if `X` is already a
 * struct tag — the typedef and tag namespaces are distinct. If the system
 * header already provided the typedef this would be a redefinition; gate
 * with a compatibility check.
 */

#ifndef OPENCODECS_HEIF_SHIM_H
#define OPENCODECS_HEIF_SHIM_H

#include <libheif/heif.h>

/* Mac's libheif (Homebrew) typedefs these as part of <libheif/heif.h>; on
 * Ubuntu they're forward-declared as `struct heif_X` only. We can detect
 * by checking for a sentinel macro that Mac's header sets, but easier:
 * #if !defined(__has_typedef) we just always declare (modern C allows
 * redundant typedefs of the same target type since C11).
 */
#if !defined(LIBHEIF_HAVE_TYPEDEFS)
typedef struct heif_context heif_context;
typedef struct heif_error heif_error;
typedef struct heif_image heif_image;
typedef struct heif_image_handle heif_image_handle;
typedef struct heif_decoding_options heif_decoding_options;
typedef struct heif_encoder heif_encoder;
typedef struct heif_encoding_options heif_encoding_options;
typedef struct heif_writer heif_writer;
#endif

#endif /* OPENCODECS_HEIF_SHIM_H */
