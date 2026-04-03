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
#include <math.h>

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
 * POLARQUANT 4-BIT — Google ICLR 2026 style (NEON for Orin Nano)
 * Block size 256 (power-of-2 for FWHT). b=4 now means PolarQuant.
 * wht_seed stores IEEE bits of per-block scale (absmax).
 * ================================================================ */

#define TQ_POLAR_BLOCK_SIZE 256u

static inline float tq_get_scale(const tq_tensor_t *t) {
    uint64_t bits = t->wht_seed;
    float scale;
    memcpy(&scale, &bits, sizeof(scale));
    return scale != 0.0f ? scale : 1.0f;
}

#if defined(__ARM_NEON) && defined(TQ_WITH_NEON)
#  include <arm_neon.h>

/* Alignment for stack arrays - GCC/Clang extension */
#  ifndef TQ_ALIGN
#    define TQ_ALIGN(x) __attribute__((aligned(x)))
#  endif

/* Fast Walsh-Hadamard Transform (in-place, NEON, size 256) */
static void tq_fwht_neon(float *restrict x) {  /* x must be _Alignas(64) */
    /* Iterative radix-2 FWHT — 8 stages for 256 */
    for (uint32_t len = 1; len < TQ_POLAR_BLOCK_SIZE; len <<= 1) {
        for (uint32_t i = 0; i < TQ_POLAR_BLOCK_SIZE; i += 2 * len) {
            for (uint32_t j = 0; j < len; j += 4) {
                float32x4_t u = vld1q_f32(&x[i + j]);
                float32x4_t v = vld1q_f32(&x[i + j + len]);
                vst1q_f32(&x[i + j],       vaddq_f32(u, v));
                vst1q_f32(&x[i + j + len], vsubq_f32(u, v));
            }
        }
    }
    /* Normalise by 1/sqrt(N) — done once at end */
    float32x4_t norm = vdupq_n_f32(1.0f / sqrtf((float)TQ_POLAR_BLOCK_SIZE));
    for (uint32_t i = 0; i < TQ_POLAR_BLOCK_SIZE; i += 4) {
        float32x4_t v = vld1q_f32(&x[i]);
        vst1q_f32(&x[i], vmulq_f32(v, norm));
    }
}

/* 4-bit lookup table (Gaussian-matched centroids, scaled later) */
static const float tq_polar_centroids[16] = {
    -2.732f, -1.931f, -1.512f, -1.194f, -0.932f, -0.707f, -0.507f, -0.324f,
     0.324f,  0.507f,  0.707f,  0.932f,  1.194f,  1.512f,  1.931f,  2.732f
};

static void tq_dequant_raw_polar4_neon(const tq_tensor_t *t,
                                       const uint8_t *restrict src,
                                       float *restrict dst) {
    const float scale = tq_get_scale(t);
    const float32x4_t v_scale = vdupq_n_f32(scale);

    uint64_t n_elements = (uint64_t)t->rows * (uint64_t)t->cols;
    uint64_t n_blocks = (n_elements + TQ_POLAR_BLOCK_SIZE - 1) / TQ_POLAR_BLOCK_SIZE;
    uint64_t i = 0;

    for (uint64_t b = 0; b < n_blocks; ++b) {
        TQ_ALIGN(64) float block[TQ_POLAR_BLOCK_SIZE];

        /* Unpack 4-bit indices (2 per byte) */
        for (uint32_t k = 0; k < TQ_POLAR_BLOCK_SIZE && i + k < n_elements; k += 2) {
            uint8_t byte = src[b * (TQ_POLAR_BLOCK_SIZE / 2) + k / 2];
            uint8_t idx0 = byte & 0x0F;
            uint8_t idx1 = byte >> 4;
            block[k]   = tq_polar_centroids[idx0];
            block[k+1] = tq_polar_centroids[idx1];
        }

        /* Inverse FWHT + scale */
        tq_fwht_neon(block);
        uint32_t block_len = (i + TQ_POLAR_BLOCK_SIZE <= n_elements) ? TQ_POLAR_BLOCK_SIZE : (uint32_t)(n_elements - i);
        for (uint32_t k = 0; k < block_len; k += 4) {
            float32x4_t v = vld1q_f32(&block[k]);
            vst1q_f32(&dst[i + k], vmulq_f32(v, v_scale));
        }
        i += TQ_POLAR_BLOCK_SIZE;
    }
}

/* Quantizer — PolarQuant 4-bit (NEON max-abs + FWHT) */
static void quantize_f32_to_polar4(const float *restrict src, uint8_t *restrict dst,
                                   uint64_t n_elements, tq_tensor_t *td) {
    td->b = 4;
    td->unpacked_size = ((n_elements + 1) / 2);  /* 4 bits → 2 per byte */

    uint64_t n_blocks = (n_elements + TQ_POLAR_BLOCK_SIZE - 1) / TQ_POLAR_BLOCK_SIZE;
    float global_max_abs = 0.0f;

    /* NEON max-abs reduction */
    float32x4_t vmax = vdupq_n_f32(0.0f);
    for (uint64_t j = 0; j + 4 <= n_elements; j += 4) {
        float32x4_t v = vld1q_f32(&src[j]);
        vmax = vmaxq_f32(vmax, vabsq_f32(v));
    }
    float32x2_t vmax2 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
    global_max_abs = fmaxf(vget_lane_f32(vmax2, 0), vget_lane_f32(vmax2, 1));

    /* scalar tail for max_abs */
    for (uint64_t j = n_elements & ~3ULL; j < n_elements; ++j) {
        float a = fabsf(src[j]);
        if (a > global_max_abs) global_max_abs = a;
    }

    /* Per-block quant (scale stored once per tensor in wht_seed) */
    uint64_t scale_bits;
    memcpy(&scale_bits, &global_max_abs, sizeof(global_max_abs));
    td->wht_seed = scale_bits;

    memset(dst, 0, (size_t)td->unpacked_size);

    TQ_ALIGN(64) float block[TQ_POLAR_BLOCK_SIZE];
    for (uint64_t b = 0; b < n_blocks; ++b) {
        uint64_t start = b * TQ_POLAR_BLOCK_SIZE;
        uint64_t len = (start + TQ_POLAR_BLOCK_SIZE <= n_elements) ? TQ_POLAR_BLOCK_SIZE : n_elements - start;

        memcpy(block, src + start, len * sizeof(float));
        if (len < TQ_POLAR_BLOCK_SIZE) memset(block + len, 0, (TQ_POLAR_BLOCK_SIZE - len) * sizeof(float));

        tq_fwht_neon(block);

        /* Scalar quantization to centroids */
        for (uint32_t k = 0; k < TQ_POLAR_BLOCK_SIZE && start + k < n_elements; k += 2) {
            float v0 = block[k];
            float v1 = block[k + 1];
            uint8_t idx0 = 0, idx1 = 0;
            float min_dist = 1e9f;
            for (uint8_t c = 0; c < 16; ++c) {
                float d = fabsf(v0 - tq_polar_centroids[c]);
                if (d < min_dist) { min_dist = d; idx0 = c; }
            }
            min_dist = 1e9f;
            for (uint8_t c = 0; c < 16; ++c) {
                float d = fabsf(v1 - tq_polar_centroids[c]);
                if (d < min_dist) { min_dist = d; idx1 = c; }
            }
            dst[b * (TQ_POLAR_BLOCK_SIZE / 2) + k / 2] = (uint8_t)(idx0 | (idx1 << 4));
        }
    }
}
#endif /* __ARM_NEON && TQ_WITH_NEON */

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
#if defined(__ARM_NEON) && defined(TQ_WITH_NEON)
    if (t->b == 4) {
        tq_dequant_raw_polar4_neon(t, src, dst);
        return;
    }
#endif

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
