/* Reading many files, or byte ranges of them, without the GIL.
 *
 * A directory of chunk files (a zarr array on local disk) is thousands of
 * open/fstat/read/close sequences. From Python every one of those calls
 * releases and retakes the GIL, and with several reader threads the
 * handoffs, not the reads, set the pace: 4096 files of 64 KiB read in
 * 52 ms on one thread, 98 ms on eight and 108 ms on sixteen, where eight
 * processes took 11 ms. Here the whole list is read in one call.
 *
 * The four file primitives and the path conversion are the only code that
 * differs by platform, all in the one block below: Windows opens a path
 * as UTF-16 (as os.open does there) and has no pread, so a read seeks
 * first; elsewhere a path is bytes in the filesystem encoding and reads
 * are positioned. Paths are converted while the GIL is held.
 */
#ifndef OC_READFILES_H
#define OC_READFILES_H

#include <Python.h>
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifdef _WIN32
#include <io.h>
#include <wchar.h>

/* ``obj`` (str, bytes or os.PathLike) as the string the OS opens. *owner
 * receives what oc_path_release frees. NULL with an exception set. */
static const void* oc_path_acquire(PyObject* obj, void** owner)
{
    Py_ssize_t n = 0;
    wchar_t* wide;
    PyObject* fs = PyOS_FSPath(obj);
    *owner = NULL;
    if (fs == NULL)
        return NULL;
    if (PyBytes_Check(fs)) {
        PyObject* text = PyUnicode_DecodeFSDefaultAndSize(
            PyBytes_AS_STRING(fs), PyBytes_GET_SIZE(fs));
        Py_DECREF(fs);
        if (text == NULL)
            return NULL;
        fs = text;
    }
    wide = PyUnicode_AsWideCharString(fs, &n);
    Py_DECREF(fs);
    if (wide == NULL)
        return NULL;
    if ((size_t) n != wcslen(wide)) {
        PyMem_Free(wide);
        PyErr_SetString(PyExc_ValueError, "embedded null character in path");
        return NULL;
    }
    *owner = wide;
    return wide;
}

static void oc_path_release(void* owner) { PyMem_Free(owner); }

static int oc_file_open(const void* path)
{
    return _wopen((const wchar_t*) path, _O_RDONLY | _O_BINARY | _O_NOINHERIT);
}

static void oc_file_close(int fd) { _close(fd); }

static int64_t oc_file_size(int fd)
{
    struct _stat64 st;
    return _fstat64(fd, &st) != 0 ? -1 : (int64_t) st.st_size;
}

/* Up to ``want`` bytes at ``offset``; fewer only at the end of the file.
 * -1 with errno set on failure. */
static int64_t oc_file_read_at(int fd, uint8_t* dst, int64_t want, int64_t offset)
{
    if (_lseeki64(fd, offset, SEEK_SET) < 0)
        return -1;
    return _read(fd, dst, (unsigned int) want);
}

#else
#include <unistd.h>

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

static const void* oc_path_acquire(PyObject* obj, void** owner)
{
    PyObject* encoded;
    PyObject* fs = PyOS_FSPath(obj);
    *owner = NULL;
    if (fs == NULL)
        return NULL;
    if (PyBytes_Check(fs)) {
        encoded = fs;
    } else {
        encoded = PyUnicode_EncodeFSDefault(fs);
        Py_DECREF(fs);
        if (encoded == NULL)
            return NULL;
    }
    if ((size_t) PyBytes_GET_SIZE(encoded) != strlen(PyBytes_AS_STRING(encoded))) {
        Py_DECREF(encoded);
        PyErr_SetString(PyExc_ValueError, "embedded null byte");
        return NULL;
    }
    *owner = encoded;
    return PyBytes_AS_STRING(encoded);
}

static void oc_path_release(void* owner) { Py_XDECREF((PyObject*) owner); }

static int oc_file_open(const void* path)
{
    int fd;
    do {
        fd = open((const char*) path, O_RDONLY | O_CLOEXEC);
    } while (fd < 0 && errno == EINTR);
    return fd;
}

static void oc_file_close(int fd) { close(fd); }

static int64_t oc_file_size(int fd)
{
    struct stat st;
    return fstat(fd, &st) != 0 ? -1 : (int64_t) st.st_size;
}

static int64_t oc_file_read_at(int fd, uint8_t* dst, int64_t want, int64_t offset)
{
    ssize_t got;
    do {
        got = pread(fd, dst, (size_t) want, (off_t) offset);
    } while (got < 0 && errno == EINTR);
    return (int64_t) got;
}
#endif


/* Up to ``n`` bytes at ``offset``, looping on short reads; fewer only at
 * the end of the file. The count read, or -1 with errno set. */
static int64_t oc_file_pread(int fd, uint8_t* dst, int64_t n, int64_t offset)
{
    int64_t done = 0, got;
    while (done < n) {
        int64_t left = n - done;
        got = oc_file_read_at(fd, dst + done, left > (1 << 30) ? (1 << 30) : left,
                              offset + done);
        if (got < 0)
            return -1;
        if (got == 0)
            break;
        done += got;
    }
    return done;
}

#endif
