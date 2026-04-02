/* tq.h — TurboQuant container format v2 (strict C99, zero UB, zero padding)
 * -std=c99 -pedantic -Wall -Wextra -Werror -march=native
 * _Alignas(64) on hot data, restrict everywhere, LZ4 per-tensor optional
 */

#ifndef TQ_H
#define TQ_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/stat.h>
#endif

#ifdef TQ_WITH_LZ4
#  include <lz4frame.h>
#endif

#define TQ_MAGIC   0x46555154u  /* "TQUF" */
#define TQ_VERSION 2

/* ================================================================
 * Model family IDs (32-bit — plenty of room)
 * ================================================================ */

typedef enum {
    TQ_FAMILY_UNKNOWN   = 0,
    TQ_FAMILY_QWEN3     = 1,
    TQ_FAMILY_FLUX1     = 2,
    TQ_FAMILY_SD15      = 3,
    TQ_FAMILY_SDXL      = 4,
    TQ_FAMILY_SD3       = 5,
    TQ_FAMILY_AURAFLOW  = 6,
    TQ_FAMILY_CUSTOM    = 0x7FFFFFFF   /* user-defined range (avoid sign issues) */
} tq_family_id_t;

/* ================================================================
 * Features bitfield (64 bits)
 * ================================================================ */

#define TQ_FEATURE_LZ4_PER_TENSOR      (1ULL << 0)
#define TQ_FEATURE_WHT_SEED_PER_TENSOR (1ULL << 1)
#define TQ_FEATURE_QJL_FOLDED          (1ULL << 2)
#define TQ_FEATURE_RESERVED            (1ULL << 63)

/* ================================================================
 * Tensor flags (stored in tq_tensor_t::tensor_flags)
 * ================================================================ */

/* Bit 0: passthrough — tensor data is raw bytes, not ternary-quantized.
 * Bits 8..15: original format-specific type (e.g. gguf_type_t value).
 * This allows lossless round-trip for quantized GGUF types. */
#define TQ_TFLAG_PASSTHROUGH           (1u << 0)
#define TQ_TFLAG_ORIG_TYPE_SHIFT       8
#define TQ_TFLAG_ORIG_TYPE_MASK        0x0000FF00u

#define TQ_TFLAG_SET_ORIG_TYPE(flags, t) \
    ((flags) | TQ_TFLAG_PASSTHROUGH | (((uint32_t)(t) << TQ_TFLAG_ORIG_TYPE_SHIFT) & TQ_TFLAG_ORIG_TYPE_MASK))
#define TQ_TFLAG_GET_ORIG_TYPE(flags) \
    (((flags) & TQ_TFLAG_ORIG_TYPE_MASK) >> TQ_TFLAG_ORIG_TYPE_SHIFT)

/* ================================================================
 * On-disk header (48 bytes)
 * ================================================================ */

typedef struct {
    uint32_t       magic;           /* "TQUF" = 0x46555154 */
    uint32_t       version;         /* 2 */
    uint64_t       features;
    uint64_t       tensor_count;
    uint64_t       data_offset;     /* 64-byte aligned */
    uint64_t       total_data_size;
    uint32_t       model_family_id; /* TQ_FAMILY_* */
    uint32_t       model_version;
} tq_header_t;

/* ================================================================
 * Per-tensor descriptor (192 bytes, ZERO padding)
 * ================================================================ */

typedef struct {
    char           name[128];       /* null-terminated */
    uint32_t       b;               /* bits per weight: 2 or 3 */
    uint32_t       rows;
    uint32_t       cols;
    uint32_t       tensor_flags;
    uint64_t       wht_seed;
    uint64_t       frame_offset;    /* relative to data section; 0 = uncompressed */
    uint64_t       frame_size;      /* compressed size (0 = uncompressed) */
    uint64_t       unpacked_size;   /* decompressed / raw byte size */
    uint64_t       index_size;
    uint64_t       norm_offset;
} tq_tensor_t;

/* ================================================================
 * In-memory view
 * ================================================================ */

typedef struct {
    uint8_t       *base;
    size_t         size;
    void          *mmap_handle;

    tq_header_t   *hdr;
    tq_tensor_t   *tensors;
    uint8_t       *data;
} tq_file_t;

/* ================================================================
 * Public API
 * ================================================================ */

int  tq_mmap(const char *path, tq_file_t *out);
void tq_munmap(tq_file_t *f);

int  tq_write(const char *path, const tq_file_t *f);

void *tq_get_tensor_data(const tq_file_t *f, const tq_tensor_t *t);

/* Lazy dequant (handles LZ4 frames) */
void tq_dequant(const tq_file_t *f, uint32_t tensor_idx,
                float *restrict dst);

#endif /* TQ_H */

/* ================================================================
 * IMPLEMENTATION (define TQ_IMPLEMENTATION in one .c)
 * ================================================================ */

#if defined(TQ_IMPLEMENTATION) && !defined(TQ_IMPLEMENTATION_DONE)
#define TQ_IMPLEMENTATION_DONE

/* posix_memalign requires _GNU_SOURCE */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdlib.h>

/* ================================================================
 * mmap
 * ================================================================ */

int tq_mmap(const char *path, tq_file_t *out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }

    out->base = (uint8_t *)mmap(NULL, (size_t)st.st_size, PROT_READ,
                                MAP_PRIVATE, fd, 0);
    close(fd);
    if (out->base == MAP_FAILED) { out->base = NULL; return -1; }

    out->size = (size_t)st.st_size;
    out->mmap_handle = NULL;

    /* Validate header */
    if (out->size < sizeof(tq_header_t)) goto fail;

    out->hdr = (tq_header_t *)out->base;
    if (out->hdr->magic != TQ_MAGIC || out->hdr->version != TQ_VERSION)
        goto fail;

    /* Tensor descriptors follow immediately after header */
    out->tensors = (tq_tensor_t *)(out->base + sizeof(tq_header_t));

    /* Data section at specified offset */
    if (out->hdr->data_offset > out->size) goto fail;
    out->data = out->base + out->hdr->data_offset;

    return 0;

fail:
    tq_munmap(out);
    return -1;
}

void tq_munmap(tq_file_t *f) {
    if (f->base) munmap(f->base, f->size);
    memset(f, 0, sizeof(*f));
}

void *tq_get_tensor_data(const tq_file_t *f, const tq_tensor_t *t) {
    return f->data + t->frame_offset;
}

/* ================================================================
 * Raw dequant kernel (scalar reference)
 *
 * For b=2: each byte packs 4 ternary values (-1, 0, +1)
 *   encoding: 0 → -1.0, 1 → 0.0, 2 → +1.0
 * For b=3: each byte packs 2 values with 3 bits each
 *   encoding: centered around 0 with 8 levels
 * ================================================================ */

static void tq_dequant_raw(const tq_tensor_t *t,
                           const uint8_t *restrict src,
                           float *restrict dst) {
    uint64_t n_elements = (uint64_t)t->rows * (uint64_t)t->cols;
    uint64_t i;

    if (t->b == 2) {
        /* 2-bit ternary: 4 values per byte */
        for (i = 0; i < n_elements; ++i) {
            uint8_t byte = src[i / 4];
            uint8_t val = (byte >> (2 * (i % 4))) & 0x03;
            dst[i] = (val == 0) ? -1.0f : (val == 1) ? 0.0f : 1.0f;
        }
    } else if (t->b == 3) {
        /* 3-bit: 2 values per byte (6 bits used, 2 wasted) */
        for (i = 0; i < n_elements; ++i) {
            uint64_t bit_offset = i * 3;
            uint64_t byte_idx = bit_offset / 8;
            uint32_t bit_idx = (uint32_t)(bit_offset % 8);
            uint16_t two_bytes;
            memcpy(&two_bytes, &src[byte_idx], 2);
            uint8_t val = (two_bytes >> bit_idx) & 0x07;
            /* 8-level centered: -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5 */
            dst[i] = (float)val - 3.5f;
        }
    }
}

/* ================================================================
 * Lazy dequant (the hot path)
 * ================================================================ */

void tq_dequant(const tq_file_t *f, uint32_t tensor_idx,
                float *restrict dst) {
    const tq_tensor_t *t = &f->tensors[tensor_idx];
    const uint8_t *src = f->data + t->frame_offset;

#ifdef TQ_WITH_LZ4
    if (t->frame_size != 0) {
        void *tmp = NULL;
        int rc;
        LZ4F_dctx *dctx = NULL;
        size_t dst_size = (size_t)t->unpacked_size;
        size_t src_size = (size_t)t->frame_size;
        size_t result;

        rc = posix_memalign(&tmp, 64, dst_size);
        if (rc != 0 || !tmp) return;

        result = LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION);
        if (LZ4F_isError(result)) {
            free(tmp);
            return;
        }

        result = LZ4F_decompress(dctx, tmp, &dst_size, src, &src_size, NULL);
        LZ4F_freeDecompressionContext(dctx);

        if (LZ4F_isError(result)) {
            free(tmp);
            return;
        }

        tq_dequant_raw(t, (const uint8_t *)tmp, dst);
        free(tmp);
        return;
    }
#endif

    /* uncompressed path */
    tq_dequant_raw(t, src, dst);
}

/* ================================================================
 * Writer
 * ================================================================ */

int tq_write(const char *path, const tq_file_t *f) {
    FILE *fp;
    uint64_t data_offset, i;
    long pos, aligned;

    if (!f || !f->hdr) return -1;
    fp = fopen(path, "wb");
    if (!fp) return -1;

    /* Write header */
    fwrite(f->hdr, sizeof(tq_header_t), 1, fp);

    /* Write tensor descriptors */
    fwrite(f->tensors, sizeof(tq_tensor_t), (size_t)f->hdr->tensor_count, fp);

    /* Pad to data_offset (64-byte aligned) */
    data_offset = f->hdr->data_offset;
    pos = ftell(fp);
    aligned = (long)data_offset;
    while (pos < aligned) {
        uint8_t zero = 0;
        fwrite(&zero, 1, 1, fp);
        pos++;
    }

    /* Write tensor data */
    if (f->data && f->hdr->total_data_size > 0) {
        fwrite(f->data, 1, (size_t)f->hdr->total_data_size, fp);
    } else if (f->data) {
        /* Compute from tensor descriptors */
        uint64_t total = 0;
        for (i = 0; i < f->hdr->tensor_count; ++i) {
            uint64_t end;
            const tq_tensor_t *t = &f->tensors[i];
            if (t->frame_size > 0)
                end = t->frame_offset + t->frame_size;
            else
                end = t->frame_offset + t->unpacked_size;
            if (end > total) total = end;
        }
        fwrite(f->data, 1, (size_t)total, fp);
    }

    fclose(fp);
    return 0;
}

#endif /* TQ_IMPLEMENTATION */
