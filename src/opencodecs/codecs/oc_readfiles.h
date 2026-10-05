/* Reading many files, or byte ranges of them, without the GIL.
 *
 * A directory of chunk files (a zarr array on local disk) is thousands of
 * open/fstat/read/close sequences. From Python every one of those calls
 * releases and retakes the GIL, and with several reader threads the
 * handoffs, not the reads, set the pace: 4096 files of 64 KiB read in
 * 52 ms on one thread, 98 ms on eight and 108 ms on sixteen, where eight
 * processes took 11 ms. Here the whole list is read in one call.
 *
 * Paths are resolved while the GIL is held (oc_path_acquire), to the
 * filesystem encoding on POSIX and to UTF-16 on Windows, as os.open does.
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
#else
#include <unistd.h>
#endif

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

/* ``obj`` (str, bytes or os.PathLike) as the string the OS opens. *owner
 * receives what oc_path_release frees. NULL with an exception set. */
static const void* oc_path_acquire(PyObject* obj, void** owner)
{
    PyObject* fs = PyOS_FSPath(obj);
    *owner = NULL;
    if (fs == NULL)
        return NULL;
#ifdef _WIN32
    {
        Py_ssize_t n = 0;
        wchar_t* wide;
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
#else
    {
        PyObject* encoded;
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
#endif
}

static void oc_path_release(void* owner)
{
    if (owner == NULL)
        return;
#ifdef _WIN32
    PyMem_Free(owner);
#else
    Py_DECREF((PyObject*) owner);
#endif
}

/* Open read-only; -1 with errno set on failure. No GIL needed. */
static int oc_file_open(const void* path)
{
#ifdef _WIN32
    return _wopen((const wchar_t*) path, _O_RDONLY | _O_BINARY | _O_NOINHERIT);
#else
    int fd;
    do {
        fd = open((const char*) path, O_RDONLY | O_CLOEXEC);
    } while (fd < 0 && errno == EINTR);
    return fd;
#endif
}

static void oc_file_close(int fd)
{
#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif
}

/* The file's size, or -1 with errno set. */
static int64_t oc_file_size(int fd)
{
#ifdef _WIN32
    struct _stat64 st;
    if (_fstat64(fd, &st) != 0)
        return -1;
    return (int64_t) st.st_size;
#else
    struct stat st;
    if (fstat(fd, &st) != 0)
        return -1;
    return (int64_t) st.st_size;
#endif
}

/* Up to ``n`` bytes at ``offset``, looping on short reads; fewer only at
 * the end of the file. The count read, or -1 with errno set. */
static int64_t oc_file_pread(int fd, uint8_t* dst, int64_t n, int64_t offset)
{
    int64_t done = 0;
#ifdef _WIN32
    if (_lseeki64(fd, offset, SEEK_SET) < 0)
        return -1;
    while (done < n) {
        int64_t left = n - done;
        int got = _read(fd, dst + done, (unsigned int) (left > (1 << 30) ? (1 << 30) : left));
        if (got < 0)
            return -1;
        if (got == 0)
            break;
        done += got;
    }
#else
    while (done < n) {
        int64_t left = n - done;
        ssize_t got = pread(fd, dst + done, (size_t) (left > (1 << 30) ? (1 << 30) : left),
                            (off_t) (offset + done));
        if (got < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (got == 0)
            break;
        done += got;
    }
#endif
    return done;
}

#endif
