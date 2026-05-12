// C-callable shim around OpenJPH's C++ ojph::codestream API.
// See openjph_shim.h for the C contract.

#include "openjph_shim.h"

#include <openjph/ojph_arch.h>
#include <openjph/ojph_base.h>
#include <openjph/ojph_mem.h>
#include <openjph/ojph_codestream.h>
#include <openjph/ojph_file.h>
#include <openjph/ojph_params.h>
#include <openjph/ojph_message.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

// Thread-local error message buffer. Plain `static thread_local` is
// sufficient: opencodecs Cython modules release the GIL only inside
// `nogil` blocks we don't enter here.
thread_local std::string g_last_error;

void set_error(const char* where, const std::exception& e) {
    g_last_error = std::string(where) + ": " + e.what();
}

void set_error(const char* msg) {
    g_last_error = msg;
}

// Encode: load one component row into line_buf->i32 as raw values.
// OpenJPH does DC-shifting internally based on the is_signed flag in
// the SIZ marker; we must NOT pre-shift. ppm_in::read in the OpenJPH
// tree does exactly this for unsigned, and signed input is passed
// through sign-extended.
template <typename T>
inline void copy_row_signed_unsigned(
    const T* src_row, ojph::si32* dst, ojph::ui32 width,
    bool is_signed_data
) {
    if (is_signed_data) {
        const auto* s = reinterpret_cast<
            const typename std::make_signed<T>::type*>(src_row);
        for (ojph::ui32 i = 0; i < width; ++i)
            dst[i] = static_cast<ojph::si32>(s[i]);
    } else {
        for (ojph::ui32 i = 0; i < width; ++i)
            dst[i] = static_cast<ojph::si32>(src_row[i]);
    }
}

// Decode: pull si32 values, clamp into the legal range, and cast.
// Matches gen_cvrt_32b1c_to_{8ub,16ub} from OpenJPH's reference tools.
template <typename T>
inline void copy_row_to_planar(
    const ojph::si32* src, T* dst_row, ojph::ui32 width,
    ojph::si32 min_val, ojph::si32 max_val
) {
    for (ojph::ui32 i = 0; i < width; ++i) {
        ojph::si32 v = src[i];
        if (v < min_val) v = min_val;
        if (v > max_val) v = max_val;
        dst_row[i] = static_cast<T>(v);
    }
}

}  // namespace

extern "C" {

const char* opencodecs_htj2k_last_error(void) {
    return g_last_error.c_str();
}

void opencodecs_htj2k_free(void* buf) {
    std::free(buf);
}

int opencodecs_htj2k_encode(
    const void* src,
    int width,
    int height,
    int components,
    int bit_depth,
    int is_signed_in,
    int bytes_per_sample,
    int reversible,
    float irrev_delta,
    int num_decomp,
    void** out_buf,
    size_t* out_size
) {
    if (!src || !out_buf || !out_size) {
        set_error("null arg");
        return 1;
    }
    if (width <= 0 || height <= 0 || components <= 0 ||
        bit_depth < 1 || bit_depth > 16 ||
        (bytes_per_sample != 1 && bytes_per_sample != 2)) {
        set_error("invalid frame info");
        return 2;
    }
    *out_buf = nullptr;
    *out_size = 0;

    const bool is_signed_data = (is_signed_in != 0);
    const size_t plane_samples = static_cast<size_t>(width) * height;

    try {
        ojph::codestream cs;
        ojph::mem_outfile mf;
        mf.open();

        ojph::param_siz siz = cs.access_siz();
        siz.set_image_extent(ojph::point(width, height));
        siz.set_num_components(components);
        for (int c = 0; c < components; ++c) {
            siz.set_component(
                c, ojph::point(1, 1),
                static_cast<ojph::ui32>(bit_depth),
                is_signed_data);
        }

        ojph::param_cod cod = cs.access_cod();
        cod.set_num_decomposition(num_decomp);
        cod.set_block_dims(64, 64);
        cod.set_reversible(reversible != 0);
        // Color transform (RGB->YCbCr) only when 3-component reversible
        // encoding is desired; safer to leave off for arbitrary inputs.
        cod.set_color_transform(false);

        if (reversible == 0) {
            ojph::param_qcd qcd = cs.access_qcd();
            qcd.set_irrev_quant(irrev_delta);
        }

        // For one-component or planar multi-component, request planar
        // exchange order (encoder pulls all rows of component 0, then
        // component 1, ...).
        cs.set_planar(true);

        cs.write_headers(&mf);

        ojph::ui32 next_comp = 0;
        ojph::line_buf* cur_line = cs.exchange(nullptr, next_comp);

        const ojph::ui32 W = static_cast<ojph::ui32>(width);
        const ojph::ui32 H = static_cast<ojph::ui32>(height);

        for (int c = 0; c < components; ++c) {
            if (static_cast<int>(next_comp) != c) {
                set_error("component order mismatch");
                return 3;
            }
            const uint8_t* src_bytes =
                reinterpret_cast<const uint8_t*>(src) +
                static_cast<size_t>(c) * plane_samples * bytes_per_sample;

            for (ojph::ui32 r = 0; r < H; ++r) {
                if (bytes_per_sample == 1) {
                    const uint8_t* row =
                        src_bytes + static_cast<size_t>(r) * W;
                    copy_row_signed_unsigned<uint8_t>(
                        row, cur_line->i32, W, is_signed_data);
                } else {
                    const uint16_t* row =
                        reinterpret_cast<const uint16_t*>(src_bytes) +
                        static_cast<size_t>(r) * W;
                    copy_row_signed_unsigned<uint16_t>(
                        row, cur_line->i32, W, is_signed_data);
                }
                cur_line = cs.exchange(cur_line, next_comp);
            }
        }
        cs.flush();
        cs.close();

        // Copy memfile bytes into a malloc'd buffer for the Python side.
        const size_t n = mf.get_used_size();
        void* buf = std::malloc(n);
        if (!buf) {
            set_error("malloc failed");
            return 4;
        }
        std::memcpy(buf, mf.get_data(), n);
        *out_buf = buf;
        *out_size = n;
        return 0;
    } catch (const std::exception& e) {
        set_error("encode", e);
        return 5;
    } catch (...) {
        set_error("encode: unknown C++ exception");
        return 5;
    }
}

int opencodecs_htj2k_decode_info(
    const void* src,
    size_t srcsize,
    int* width,
    int* height,
    int* components,
    int* bit_depth,
    int* is_signed_out
) {
    if (!src || srcsize == 0 || !width || !height || !components ||
        !bit_depth || !is_signed_out) {
        set_error("null arg");
        return 1;
    }
    try {
        ojph::codestream cs;
        ojph::mem_infile mf;
        mf.open(reinterpret_cast<const ojph::ui8*>(src), srcsize);
        cs.read_headers(&mf);

        ojph::param_siz siz = cs.access_siz();
        ojph::point ext = siz.get_image_extent();
        *width = static_cast<int>(ext.x);
        *height = static_cast<int>(ext.y);
        *components = static_cast<int>(siz.get_num_components());
        *bit_depth = static_cast<int>(siz.get_bit_depth(0));
        *is_signed_out = siz.is_signed(0) ? 1 : 0;
        cs.close();
        return 0;
    } catch (const std::exception& e) {
        set_error("decode_info", e);
        return 2;
    } catch (...) {
        set_error("decode_info: unknown C++ exception");
        return 2;
    }
}

int opencodecs_htj2k_decode(
    const void* src,
    size_t srcsize,
    void* dst,
    size_t dst_size,
    int bytes_per_sample
) {
    if (!src || srcsize == 0 || !dst) {
        set_error("null arg");
        return 1;
    }
    if (bytes_per_sample != 1 && bytes_per_sample != 2) {
        set_error("invalid bytes_per_sample");
        return 2;
    }
    try {
        ojph::codestream cs;
        ojph::mem_infile mf;
        mf.open(reinterpret_cast<const ojph::ui8*>(src), srcsize);
        cs.read_headers(&mf);

        ojph::param_siz siz = cs.access_siz();
        ojph::point ext = siz.get_image_extent();
        const ojph::ui32 W = ext.x;
        const ojph::ui32 H = ext.y;
        const int comps = static_cast<int>(siz.get_num_components());
        const int bd = static_cast<int>(siz.get_bit_depth(0));
        const bool sg = siz.is_signed(0);

        const size_t plane_samples = static_cast<size_t>(W) * H;
        const size_t need =
            plane_samples * comps * bytes_per_sample;
        if (dst_size < need) {
            set_error("destination buffer too small");
            return 3;
        }

        cs.set_planar(true);
        cs.create();

        const ojph::si32 min_val =
            sg ? -(ojph::si32(1) << (bd - 1)) : 0;
        const ojph::si32 max_val =
            sg ? ( (ojph::si32(1) << (bd - 1)) - 1 )
               : ( (ojph::si32(1) << bd) - 1 );

        for (int c = 0; c < comps; ++c) {
            uint8_t* dst_bytes =
                reinterpret_cast<uint8_t*>(dst) +
                static_cast<size_t>(c) * plane_samples * bytes_per_sample;
            for (ojph::ui32 r = 0; r < H; ++r) {
                ojph::ui32 pulled_comp = 0;
                ojph::line_buf* line = cs.pull(pulled_comp);
                if (static_cast<int>(pulled_comp) != c) {
                    set_error("decode: component order mismatch");
                    return 4;
                }
                if (bytes_per_sample == 1) {
                    uint8_t* row =
                        dst_bytes + static_cast<size_t>(r) * W;
                    copy_row_to_planar<uint8_t>(
                        line->i32, row, W, min_val, max_val);
                } else {
                    uint16_t* row =
                        reinterpret_cast<uint16_t*>(dst_bytes) +
                        static_cast<size_t>(r) * W;
                    copy_row_to_planar<uint16_t>(
                        line->i32, row, W, min_val, max_val);
                }
            }
        }
        cs.close();
        return 0;
    } catch (const std::exception& e) {
        set_error("decode", e);
        return 5;
    } catch (...) {
        set_error("decode: unknown C++ exception");
        return 5;
    }
}

}  // extern "C"
