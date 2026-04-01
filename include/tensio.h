/* tensio.h — Master header for the Tensio tensor I/O library
 * Strict C99, zero UB, header-only
 * MIT License — see LICENSE
 */

#ifndef TENSIO_H
#define TENSIO_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#define TENSIO_VERSION_MAJOR 0
#define TENSIO_VERSION_MINOR 1
#define TENSIO_VERSION_PATCH 0
#define TENSIO_VERSION "0.1.0"

/* ================================================================
 * I/O backend interface (used by bundle + io_uring)
 * ================================================================ */

typedef struct tensio_io_backend {
    int  (*open)(const char *path, int flags);
    int  (*read_async)(int fd, void *buf, size_t len, off_t offset,
                       void *user_data);
    void (*close)(int fd);
} tensio_io_backend_t;

/* ================================================================
 * Format headers
 * ================================================================ */

#include "safetensors.h"
#include "gguf.h"
#include "tq.h"
#include "convert.h"

#ifdef TENSIO_ENABLE_VTABLES
#  include "sqlite/vtable_safetensors.h"
#  include "sqlite/vtable_gguf.h"
#  include "sqlite/vtable_tq.h"
#endif

#ifdef TENSIO_ENABLE_BUNDLE
#  include "bundle.h"
#endif

#ifdef TENSIO_ENABLE_IO_URING
#  include "io_uring.h"
#endif

#ifdef TENSIO_ENABLE_STB_IMAGE
#  define STB_IMAGE_WRITE_IMPLEMENTATION
#  include "../third_party/stb_image_write.h"
#endif

#endif /* TENSIO_H */

