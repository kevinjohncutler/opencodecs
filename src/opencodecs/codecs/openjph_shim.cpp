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

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

// Thread-local error message buffer. Plain `static thread_local` is
// sufficient: opencodecs Cython modules release the GIL only inside
// `nogil` blocks we don't enter here.
thread_local std::string g_last_error;
thread_local std::string g_error_detail;

void set_error(const char* where, const std::exception& e) {
    // Prefer the text OpenJPH formatted; e.what() is generic.
    const std::string& detail = g_error_detail.empty()
        ? std::string(e.what()) : g_error_detail;
    g_last_error = std::string(where) + ": " + detail;
}

void set_error(const char* msg) {
    g_last_error = msg;
}

// Warnings OpenJPH raised during the current call.
//
// Two reasons to intercept these rather than let the library have its
// way with them. First, the default handler prints to stdout, and a
// library writing to a process's stdout corrupts whatever that process
// was writing there. Second, and worse: several of those warnings say a
// marker segment "is not supported yet" and then decoding continues, so
// the caller is handed an image built from a codestream the decoder
// admits it did not fully read. Silent wrong pixels are the one failure
// mode worse than an exception, so we collect the text and let the
// Python layer decide.
thread_local std::string g_warnings;

class warning_collector : public ojph::message_warning {
public:
    void operator()(int, const char*, int, const char* fmt, ...) override {
        char buf[1024];
        va_list args;
        va_start(args, fmt);
        int n = vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        if (n > 0) {
            if (!g_warnings.empty()) g_warnings += "\n";
            g_warnings += buf;
        }
    }
};

// OpenJPH's default error handler prints the reason and then throws a
// generic exception, so catching it yields "ojph error" with the useful
// half left on stdout. Collecting the text here is what lets a caller
// see "this codestream has 5 quality layers" instead of "rc=2".

class error_collector : public ojph::message_error {
public:
    void operator()(int, const char*, int, const char* fmt, ...) override {
        char buf[1024];
        va_list args;
        va_start(args, fmt);
        int n = vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        g_error_detail = (n > 0) ? buf : "";
        throw std::runtime_error(g_error_detail);
    }
};

warning_collector g_warning_collector;
error_collector g_error_collector;
bool g_handlers_installed = false;

void install_warning_collector() {
    if (!g_handlers_installed) {
        ojph::configure_warning(&g_warning_collector);
        ojph::configure_error(&g_error_collector);
        g_handlers_installed = true;
    }
}

// Encode: read one row of one component into an OpenJPH line.
//
// OpenJPH does the DC level shift itself from the SIZ signedness, so
// samples go in as their plain integer values: unsigned zero-extended,
// signed sign-extended. `step` is 1 for a planar source and the
// component count for an interleaved one. A float32 image arrives as
// its int32 bit pattern; the NLT marker tells a decoder what it is.
template <typename T>
inline void load_row(const T* s, ojph::line_buf* line, ojph::ui32 width,
                     size_t step) {
    if (line->flags & ojph::line_buf::LFT_64BIT) {
        ojph::si64* d = line->i64;
        for (ojph::ui32 i = 0; i < width; ++i)
            d[i] = static_cast<ojph::si64>(s[i * step]);
    } else {
        ojph::si32* d = line->i32;
        for (ojph::ui32 i = 0; i < width; ++i)
            d[i] = static_cast<ojph::si32>(s[i * step]);
    }
}

bool load_component_row(const void* src, int bytes_per_sample,
                        bool is_signed, size_t offset, size_t step,
                        ojph::line_buf* line, ojph::ui32 width) {
    if (!(line->flags & ojph::line_buf::LFT_INTEGER)) return false;
    switch (bytes_per_sample * 2 + (is_signed ? 1 : 0)) {
    case 2: load_row(static_cast<const uint8_t*>(src) + offset, line, width, step); break;
    case 3: load_row(static_cast<const int8_t*>(src) + offset, line, width, step); break;
    case 4: load_row(static_cast<const uint16_t*>(src) + offset, line, width, step); break;
    case 5: load_row(static_cast<const int16_t*>(src) + offset, line, width, step); break;
    case 8: load_row(static_cast<const uint32_t*>(src) + offset, line, width, step); break;
    case 9: load_row(static_cast<const int32_t*>(src) + offset, line, width, step); break;
    default: return false;
    }
    return true;
}

// Decode: store one pulled line, clamped into the component's nominal
// range (ISO/IEC 15444-1 Annex G.1.2 names clipping as the usual
// treatment of quantization overshoot; OpenJPH's ojph_expand and
// OpenJPEG both clip). The arithmetic is 64-bit so a 31-bit range does
// not overflow. A 32-bit component is copied bit for bit: an unsigned
// 32-bit sample does not fit a signed 32-bit line, so OpenJPH hands it
// over in two's complement and only the bit pattern is meaningful.
template <typename T, typename L>
inline void store_row(const L* s, T* d, ojph::ui32 width, size_t step,
                      bool clamp, ojph::si64 lo, ojph::si64 hi) {
    if (clamp) {
        for (ojph::ui32 i = 0; i < width; ++i) {
            ojph::si64 v = static_cast<ojph::si64>(s[i]);
            if (v < lo) v = lo;
            if (v > hi) v = hi;
            d[i * step] = static_cast<T>(v);
        }
    } else {
        for (ojph::ui32 i = 0; i < width; ++i)
            d[i * step] = static_cast<T>(s[i]);
    }
}

template <typename T>
inline void store_line(const ojph::line_buf* line, T* d, ojph::ui32 width,
                       size_t step, bool clamp, ojph::si64 lo,
                       ojph::si64 hi) {
    if (line->flags & ojph::line_buf::LFT_64BIT)
        store_row<T, ojph::si64>(line->i64, d, width, step, clamp, lo, hi);
    else
        store_row<T, ojph::si32>(line->i32, d, width, step, clamp, lo, hi);
}

bool store_component_row(const ojph::line_buf* line, void* dst,
                         int bytes_per_sample, bool is_signed,
                         size_t offset, size_t step, ojph::ui32 width,
                         int bit_depth) {
    if (!(line->flags & ojph::line_buf::LFT_INTEGER)) return false;
    const bool clamp = bit_depth < 32 ||
        (line->flags & ojph::line_buf::LFT_64BIT) != 0;
    const ojph::si64 lo = is_signed ? -(ojph::si64(1) << (bit_depth - 1)) : 0;
    const ojph::si64 hi = is_signed ? (ojph::si64(1) << (bit_depth - 1)) - 1
                                    : (ojph::si64(1) << bit_depth) - 1;
    switch (bytes_per_sample * 2 + (is_signed ? 1 : 0)) {
    case 2: store_line(line, static_cast<uint8_t*>(dst) + offset, width, step, clamp, lo, hi); break;
    case 3: store_line(line, static_cast<int8_t*>(dst) + offset, width, step, clamp, lo, hi); break;
    case 4: store_line(line, static_cast<uint16_t*>(dst) + offset, width, step, clamp, lo, hi); break;
    case 5: store_line(line, static_cast<int16_t*>(dst) + offset, width, step, clamp, lo, hi); break;
    case 8: store_line(line, static_cast<uint32_t*>(dst) + offset, width, step, clamp, lo, hi); break;
    case 9: store_line(line, static_cast<int32_t*>(dst) + offset, width, step, clamp, lo, hi); break;
    default: return false;
    }
    return true;
}

// Header facts shared by decode_info and decode.
struct header_facts {
    int components;
    int bit_depth;
    bool is_signed;
    bool color_transform;
    int nlt_type;
    bool uniform;
};

int nlt_type_of(ojph::codestream& cs, ojph::ui32 c) {
    ojph::ui8 bd = 0, type = 0;
    bool sg = false;
    if (cs.access_nlt().get_nonlinear_transform(c, bd, sg, type))
        return static_cast<int>(type);
    return 0;
}

header_facts read_facts(ojph::codestream& cs) {
    header_facts f;
    ojph::param_siz siz = cs.access_siz();
    f.components = static_cast<int>(siz.get_num_components());
    f.bit_depth = static_cast<int>(siz.get_bit_depth(0));
    f.is_signed = siz.is_signed(0);
    f.color_transform = cs.access_cod().is_using_color_transform();
    f.nlt_type = nlt_type_of(cs, 0);
    f.uniform = true;
    for (int c = 0; c < f.components; ++c) {
        const ojph::ui32 uc = static_cast<ojph::ui32>(c);
        ojph::point ds = siz.get_downsampling(uc);
        if (static_cast<int>(siz.get_bit_depth(uc)) != f.bit_depth ||
            siz.is_signed(uc) != f.is_signed ||
            ds.x != 1 || ds.y != 1 ||
            nlt_type_of(cs, uc) != f.nlt_type) {
            f.uniform = false;
        }
    }
    return f;
}

}  // namespace

extern "C" {

const char* opencodecs_htj2k_last_error(void) {
    return g_last_error.c_str();
}

const char* opencodecs_htj2k_last_warnings(void) {
    return g_warnings.c_str();
}

void opencodecs_htj2k_clear_warnings(void) {
    g_warnings.clear();
    install_warning_collector();
}

void opencodecs_htj2k_free(void* buf) {
    std::free(buf);
}

int opencodecs_htj2k_encode(
    const void* src,
    const opencodecs_htj2k_encode_params* p,
    void** out_buf,
    size_t* out_size
) {
    install_warning_collector();
    if (!src || !p || !out_buf || !out_size) {
        set_error("null arg");
        return 1;
    }
    if (p->width <= 0 || p->height <= 0 || p->components <= 0 ||
        p->bit_depth < 1 || p->bit_depth > 32 ||
        (p->bytes_per_sample != 1 && p->bytes_per_sample != 2 &&
         p->bytes_per_sample != 4) ||
        p->bit_depth > 8 * p->bytes_per_sample) {
        set_error("invalid frame info");
        return 2;
    }
    if (p->color_transform && p->components < 3) {
        set_error("the component transform needs at least 3 components");
        return 2;
    }
    *out_buf = nullptr;
    *out_size = 0;

    const bool is_signed = (p->is_signed != 0);
    const ojph::ui32 W = static_cast<ojph::ui32>(p->width);
    const ojph::ui32 H = static_cast<ojph::ui32>(p->height);
    const ojph::ui32 C = static_cast<ojph::ui32>(p->components);
    const size_t plane = static_cast<size_t>(W) * H;

    try {
        ojph::codestream cs;
        ojph::mem_outfile mf;
        mf.open();

        if (p->profile) cs.set_profile(p->profile);
        cs.set_tilepart_divisions(p->tilepart_resolutions != 0,
                                  p->tilepart_components != 0);
        cs.request_tlm_marker(p->tlm != 0);

        ojph::param_siz siz = cs.access_siz();
        siz.set_image_extent(ojph::point(W, H));
        if (p->tile_w > 0 && p->tile_h > 0) {
            siz.set_tile_size(ojph::size(
                static_cast<ojph::ui32>(p->tile_w),
                static_cast<ojph::ui32>(p->tile_h)));
        }
        siz.set_num_components(C);
        for (ojph::ui32 c = 0; c < C; ++c) {
            siz.set_component(c, ojph::point(1, 1),
                              static_cast<ojph::ui32>(p->bit_depth),
                              is_signed);
        }

        ojph::param_cod cod = cs.access_cod();
        if (p->num_decomp >= 0)
            cod.set_num_decomposition(static_cast<ojph::ui32>(p->num_decomp));
        if (p->block_w > 0 && p->block_h > 0)
            cod.set_block_dims(static_cast<ojph::ui32>(p->block_w),
                               static_cast<ojph::ui32>(p->block_h));
        cod.set_reversible(p->reversible != 0);
        // The component transform (RCT on the reversible path, ICT on
        // the irreversible one) decorrelates components 0..2. It is
        // signaled in COD SGcod and every conforming decoder inverts it.
        cod.set_color_transform(p->color_transform != 0);
        if (p->prog_order) cod.set_progression_order(p->prog_order);

        if (p->reversible == 0) {
            ojph::param_qcd qcd = cs.access_qcd();
            if (p->qfactor > 0)
                qcd.set_qfactor(static_cast<ojph::ui8>(p->qfactor));
            else if (p->irrev_delta > 0.0f)
                qcd.set_irrev_quant(p->irrev_delta);
        }
        if (p->nlt_binary_complement) {
            cs.access_nlt().set_nonlinear_transform(
                ojph::param_nlt::ALL_COMPS,
                ojph::param_nlt::OJPH_NLT_BINARY_COMPLEMENT_NLT);
        }

        // OpenJPH can take whole components one after the other only
        // when no component transform is in use; with one, it needs one
        // row of every component at a time (ojph_codestream.h).
        const bool planar_exchange = (p->color_transform == 0);
        cs.set_planar(planar_exchange);

        cs.write_headers(&mf);

        ojph::ui32 next_comp = 0;
        ojph::line_buf* line = cs.exchange(nullptr, next_comp);

        // Where sample (c, r, 0) lives in src, and the stride between
        // neighbors in a row.
        const size_t step = p->src_planar ? 1 : C;
        auto offset_of = [&](ojph::ui32 c, ojph::ui32 r) -> size_t {
            return p->src_planar ? c * plane + static_cast<size_t>(r) * W
                                 : (static_cast<size_t>(r) * W) * C + c;
        };
        auto push = [&](ojph::ui32 c, ojph::ui32 r) -> bool {
            if (next_comp != c) {
                set_error("component order mismatch");
                return false;
            }
            if (!load_component_row(src, p->bytes_per_sample, is_signed,
                                    offset_of(c, r), step, line, W)) {
                set_error("unexpected line buffer type");
                return false;
            }
            line = cs.exchange(line, next_comp);
            return true;
        };

        if (planar_exchange) {
            for (ojph::ui32 c = 0; c < C; ++c)
                for (ojph::ui32 r = 0; r < H; ++r)
                    if (!push(c, r)) return 3;
        } else {
            for (ojph::ui32 r = 0; r < H; ++r)
                for (ojph::ui32 c = 0; c < C; ++c)
                    if (!push(c, r)) return 3;
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
    int reduce_data,
    int reduce_recon,
    int resilient,
    opencodecs_htj2k_info* info
) {
    install_warning_collector();
    if (!src || srcsize == 0 || !info) {
        set_error("null arg");
        return 1;
    }
    if (reduce_data < 0 || reduce_recon < 0) {
        set_error("reduce must be >= 0");
        return 1;
    }
    try {
        ojph::codestream cs;
        ojph::mem_infile mf;
        mf.open(reinterpret_cast<const ojph::ui8*>(src), srcsize);
        if (resilient) cs.enable_resilience();
        cs.read_headers(&mf);

        const int ndecomp =
            static_cast<int>(cs.access_cod().get_num_decompositions());
        info->num_decompositions = ndecomp;
        if (reduce_data > ndecomp || reduce_recon > ndecomp) {
            set_error("reduce exceeds the codestream's decomposition count");
            return 4;
        }

        const header_facts f = read_facts(cs);
        info->components = f.components;
        info->bit_depth = f.bit_depth;
        info->is_signed = f.is_signed ? 1 : 0;
        info->color_transform = f.color_transform ? 1 : 0;
        info->nlt_type = f.nlt_type;
        info->uniform = f.uniform ? 1 : 0;

        // get_recon_* reports the reduced geometry once the restriction
        // is in place, so ask for it the same way the decode will.
        if (reduce_data > 0 || reduce_recon > 0) {
            cs.restrict_input_resolution(
                static_cast<ojph::ui32>(reduce_data),
                static_cast<ojph::ui32>(reduce_recon));
        }
        ojph::param_siz siz = cs.access_siz();
        info->width = static_cast<int>(siz.get_recon_width(0));
        info->height = static_cast<int>(siz.get_recon_height(0));
        if (info->width <= 0 || info->height <= 0) {
            set_error("reduce leaves a zero-sized image");
            return 4;
        }
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
    int bytes_per_sample,
    int reduce_data,
    int reduce_recon,
    int resilient,
    int dst_planar
) {
    install_warning_collector();
    if (!src || srcsize == 0 || !dst) {
        set_error("null arg");
        return 1;
    }
    if (bytes_per_sample != 1 && bytes_per_sample != 2 &&
        bytes_per_sample != 4) {
        set_error("invalid bytes_per_sample");
        return 2;
    }
    if (reduce_data < 0 || reduce_recon < 0) {
        set_error("reduce must be >= 0");
        return 2;
    }
    try {
        ojph::codestream cs;
        ojph::mem_infile mf;
        mf.open(reinterpret_cast<const ojph::ui8*>(src), srcsize);
        if (resilient) cs.enable_resilience();
        cs.read_headers(&mf);

        const int ndecomp =
            static_cast<int>(cs.access_cod().get_num_decompositions());
        if (reduce_data > ndecomp || reduce_recon > ndecomp) {
            set_error("reduce exceeds the codestream's decomposition count");
            return 6;
        }
        // Must land between read_headers() and create(): OpenJPH uses it
        // to decide which subbands to read at all, so the finest
        // resolutions are never entropy-decoded rather than decoded and
        // discarded.
        if (reduce_data > 0 || reduce_recon > 0) {
            cs.restrict_input_resolution(
                static_cast<ojph::ui32>(reduce_data),
                static_cast<ojph::ui32>(reduce_recon));
        }

        const header_facts f = read_facts(cs);
        if (!f.uniform) {
            set_error("components differ in bit depth, signedness, "
                      "nonlinearity or subsampling; not supported");
            return 7;
        }
        if (f.bit_depth < 1 || f.bit_depth > 8 * bytes_per_sample) {
            set_error("bytes_per_sample too small for the bit depth");
            return 2;
        }

        ojph::param_siz siz = cs.access_siz();
        const ojph::ui32 W = siz.get_recon_width(0);
        const ojph::ui32 H = siz.get_recon_height(0);
        const ojph::ui32 C = static_cast<ojph::ui32>(f.components);

        const size_t plane = static_cast<size_t>(W) * H;
        const size_t need = plane * C * bytes_per_sample;
        if (dst_size < need) {
            set_error("destination buffer too small");
            return 3;
        }

        // A codestream that uses the component transform must be pulled
        // one row of every component at a time: the inverse RCT/ICT
        // needs components 0..2 of a row together. Pulling it planar
        // returns the transformed samples untouched, which is wrong data
        // with no error (OpenJPH documents planar as "can only be used
        // when there is no color transform").
        const bool planar_pull = !f.color_transform;
        cs.set_planar(planar_pull);
        cs.create();

        const size_t step = dst_planar ? 1 : C;
        auto offset_of = [&](ojph::ui32 c, ojph::ui32 r) -> size_t {
            return dst_planar ? c * plane + static_cast<size_t>(r) * W
                              : (static_cast<size_t>(r) * W) * C + c;
        };
        auto pull = [&](ojph::ui32 c, ojph::ui32 r) -> int {
            ojph::ui32 got = 0;
            ojph::line_buf* line = cs.pull(got);
            if (!line || got != c) {
                set_error("decode: component order mismatch");
                return 4;
            }
            if (line->size < W) {
                set_error("decode: short line");
                return 4;
            }
            if (!store_component_row(line, dst, bytes_per_sample,
                                     f.is_signed, offset_of(c, r), step,
                                     W, f.bit_depth)) {
                set_error("decode: unexpected line buffer type");
                return 4;
            }
            return 0;
        };

        int rc = 0;
        if (planar_pull) {
            for (ojph::ui32 c = 0; c < C && rc == 0; ++c)
                for (ojph::ui32 r = 0; r < H && rc == 0; ++r)
                    rc = pull(c, r);
        } else {
            for (ojph::ui32 r = 0; r < H && rc == 0; ++r)
                for (ojph::ui32 c = 0; c < C && rc == 0; ++c)
                    rc = pull(c, r);
        }
        if (rc != 0) return rc;
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
