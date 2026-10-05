/* Many independent compressed chunks, decoded in one native call.
 *
 * A chunked array (zarr chunks, TIFF strips, HDF5 chunks) is thousands of
 * small streams that decode without reference to each other. One Python
 * call per chunk costs tens of microseconds of interpreter work, all of it
 * under the GIL, so threads queue on the GIL instead of decoding: 64 KiB
 * LZ4 chunks decoded slower on 8 threads than on one. The codec modules
 * (_zstd, _deflate, _lz4) instead take a run of chunks in one call: this
 * header resolves every chunk's source and destination pointer while the
 * GIL is held, and the module's decode loop then runs over the whole run
 * without it.
 *
 * Sources are a sequence of bytes-like chunks, or one buffer that chunk
 * ``i`` spans from ``chunk_offsets[i]`` to ``chunk_offsets[i + 1]``.
 * Destinations are one writable buffer per chunk (a list or tuple), or one
 * writable buffer that chunk ``i`` fills from ``offsets[i]`` up to
 * ``offsets[i + 1]``. Offsets are C-contiguous 64-bit integer buffers
 * (numpy int64) of len(chunks) + 1 nondecreasing entries.
 */
#ifndef OC_BATCH_H
#define OC_BATCH_H

#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    Py_ssize_t n;            /* chunks in this run */
    const uint8_t** src;     /* chunk i's stored bytes (NULL when empty) */
    size_t* src_len;
    uint8_t** dst;           /* chunk i's destination (NULL when empty) */
    size_t* dst_cap;
    Py_buffer* views;        /* every buffer acquired, released together */
    Py_ssize_t n_views;
} oc_batch;


static void oc_batch_release(oc_batch* b)
{
    Py_ssize_t i;
    if (b->views != NULL) {
        for (i = 0; i < b->n_views; i++)
            PyBuffer_Release(&b->views[i]);
    }
    free(b->views);
    free((void*) b->src);
    free(b->src_len);
    free(b->dst);
    free(b->dst_cap);
    memset(b, 0, sizeof(*b));
}


/* Acquire ``obj`` into the next view slot. */
static int oc_batch_view(oc_batch* b, PyObject* obj, int flags, Py_buffer** out)
{
    Py_buffer* view = &b->views[b->n_views];
    if (PyObject_GetBuffer(obj, view, flags) < 0)
        return -1;
    b->n_views++;
    *out = view;
    return 0;
}


/* ``obj`` as ``count`` nondecreasing int64 offsets within ``limit`` bytes. */
static int oc_batch_offsets(oc_batch* b, PyObject* obj, Py_ssize_t count,
                            Py_ssize_t limit, const char* what,
                            const int64_t** values)
{
    Py_buffer* view;
    const char* fmt;
    const int64_t* v;
    Py_ssize_t i;
    if (oc_batch_view(b, obj, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS, &view) < 0)
        return -1;
    fmt = view->format ? view->format : "B";
    if (*fmt == '@' || *fmt == '=' || *fmt == '<')
        fmt++;
    if (view->itemsize != 8 || !(fmt[0] == 'q' || fmt[0] == 'l' || fmt[0] == 'n')
            || fmt[1] != '\0') {
        PyErr_Format(PyExc_TypeError, "%s must be int64, got format '%s'",
                     what, view->format ? view->format : "B");
        return -1;
    }
    if (view->len / 8 != count) {
        PyErr_Format(PyExc_ValueError, "%s needs %zd entries (one per chunk, "
                     "plus the end), got %zd", what, count, view->len / 8);
        return -1;
    }
    v = (const int64_t*) view->buf;
    for (i = 0; i + 1 < count; i++) {
        if (v[i] < 0 || v[i] > v[i + 1] || v[i + 1] > (int64_t) limit) {
            PyErr_Format(PyExc_ValueError,
                         "%s[%zd:%zd] = (%lld, %lld) does not lie within the "
                         "%zd-byte buffer", what, i, i + 2, (long long) v[i],
                         (long long) v[i + 1], limit);
            return -1;
        }
    }
    *values = v;
    return 0;
}


/* Fill ``b`` for chunks[start:stop] (see the top of this file). Returns 0,
 * or -1 with a Python exception set and nothing held. */
static int oc_batch_acquire(oc_batch* b, PyObject* chunks, PyObject* chunk_offsets,
                            PyObject* out, PyObject* offsets,
                            Py_ssize_t start, Py_ssize_t stop)
{
    PyObject* seq = NULL;
    PyObject* outs = NULL;
    Py_ssize_t total, i, n, k;
    int per_chunk, one_source = chunk_offsets != Py_None;
    Py_buffer* view;
    const uint8_t* src_base = NULL;
    uint8_t* dst_base = NULL;
    const int64_t* in_off = NULL;
    const int64_t* out_off = NULL;

    memset(b, 0, sizeof(*b));
    per_chunk = PyList_Check(out) || PyTuple_Check(out);
    if (per_chunk) {
        outs = PySequence_Fast(out, "out must be a buffer or a sequence of buffers");
        if (outs == NULL)
            return -1;
    }
    if (one_source) {
        if (per_chunk) {
            total = PySequence_Fast_GET_SIZE(outs);
        } else if (offsets != Py_None) {
            total = PyObject_Length(offsets) - 1;
            if (total < -1)
                goto fail;
        } else {
            PyErr_SetString(PyExc_ValueError,
                            "one output buffer needs offsets");
            goto fail;
        }
    } else {
        seq = PySequence_Fast(chunks, "chunks must be a sequence of bytes-like objects");
        if (seq == NULL)
            goto fail;
        total = PySequence_Fast_GET_SIZE(seq);
    }
    if (total < 0 || start < 0 || stop > total || start > stop) {
        PyErr_SetString(PyExc_ValueError, "chunk run lies outside the chunk list");
        goto fail;
    }
    if (per_chunk && PySequence_Fast_GET_SIZE(outs) != total) {
        PyErr_Format(PyExc_ValueError, "got %zd chunks but %zd output buffers",
                     total, PySequence_Fast_GET_SIZE(outs));
        goto fail;
    }
    n = stop - start;
    b->n = n;
    b->src = (const uint8_t**) calloc((size_t) n + 1, sizeof(uint8_t*));
    b->src_len = (size_t*) calloc((size_t) n + 1, sizeof(size_t));
    b->dst = (uint8_t**) calloc((size_t) n + 1, sizeof(uint8_t*));
    b->dst_cap = (size_t*) calloc((size_t) n + 1, sizeof(size_t));
    b->views = (Py_buffer*) calloc((size_t) 2 * n + 5, sizeof(Py_buffer));
    if (b->src == NULL || b->src_len == NULL || b->dst == NULL
            || b->dst_cap == NULL || b->views == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    if (one_source) {
        if (oc_batch_view(b, chunks, PyBUF_SIMPLE, &view) < 0)
            goto fail;
        src_base = (const uint8_t*) view->buf;
        if (oc_batch_offsets(b, chunk_offsets, total + 1, view->len,
                             "chunk_offsets", &in_off) < 0)
            goto fail;
    }
    if (!per_chunk) {
        if (oc_batch_view(b, out, PyBUF_WRITABLE, &view) < 0)
            goto fail;
        dst_base = (uint8_t*) view->buf;
        if (offsets == Py_None) {
            PyErr_SetString(PyExc_ValueError, "one output buffer needs offsets");
            goto fail;
        }
        if (oc_batch_offsets(b, offsets, total + 1, view->len, "offsets",
                             &out_off) < 0)
            goto fail;
    }
    for (k = 0; k < n; k++) {
        i = start + k;
        if (one_source) {
            b->src_len[k] = (size_t) (in_off[i + 1] - in_off[i]);
            b->src[k] = b->src_len[k] ? src_base + in_off[i] : NULL;
        } else {
            if (oc_batch_view(b, PySequence_Fast_GET_ITEM(seq, i), PyBUF_SIMPLE,
                              &view) < 0)
                goto fail;
            b->src_len[k] = (size_t) view->len;
            b->src[k] = view->len ? (const uint8_t*) view->buf : NULL;
        }
        if (per_chunk) {
            if (oc_batch_view(b, PySequence_Fast_GET_ITEM(outs, i),
                              PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS, &view) < 0)
                goto fail;
            b->dst_cap[k] = (size_t) view->len;
            b->dst[k] = view->len ? (uint8_t*) view->buf : NULL;
        } else {
            b->dst_cap[k] = (size_t) (out_off[i + 1] - out_off[i]);
            b->dst[k] = b->dst_cap[k] ? dst_base + out_off[i] : NULL;
        }
    }
    Py_XDECREF(outs);
    Py_XDECREF(seq);
    return 0;

fail:
    oc_batch_release(b);
    Py_XDECREF(outs);
    Py_XDECREF(seq);
    return -1;
}

#endif
