#define _POSIX_C_SOURCE 199309L
/* bench_parse.c — Parse latency benchmarks for all three formats
 *
 * Creates in-memory fixture files of configurable size, then measures
 * mmap + parse time via clock_gettime(CLOCK_MONOTONIC).
 *
 * Usage: bench_parse [num_tensors]    (default: 100)
 */

#include "safetensors.h"
#include "gguf.h"
#include "tq.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WARMUP_ITERS 2
#define BENCH_ITERS  10

static double elapsed_ms(struct timespec *start, struct timespec *end) {
    double s = (double)(end->tv_sec - start->tv_sec) * 1e3;
    double ns = (double)(end->tv_nsec - start->tv_nsec) / 1e6;
    return s + ns;
}

/* ----------------------------------------------------------------
 * Fixture generators
 * ---------------------------------------------------------------- */

static int gen_safetensors(const char *path, int n_tensors) {
    FILE *fp;
    char json[1024 * 64];
    int pos = 0;
    uint64_t data_offset = 0;
    uint64_t json_len;
    int i;
    float dummy[64]; /* 64 floats = 256 bytes per tensor */

    memset(dummy, 0, sizeof(dummy));
    pos += snprintf(json + pos, sizeof(json) - (size_t)pos, "{");

    for (i = 0; i < n_tensors; ++i) {
        uint64_t sz = 256;
        if (i > 0) pos += snprintf(json + pos, sizeof(json) - (size_t)pos, ",");
        pos += snprintf(json + pos, sizeof(json) - (size_t)pos,
            "\"t_%04d\":{\"dtype\":\"F32\",\"shape\":[8,8],"
            "\"data_offsets\":[%llu,%llu]}",
            i, (unsigned long long)data_offset,
            (unsigned long long)(data_offset + sz));
        data_offset += sz;
    }
    pos += snprintf(json + pos, sizeof(json) - (size_t)pos, "}");

    json_len = (uint64_t)pos;
    fp = fopen(path, "wb");
    if (!fp) return -1;
    fwrite(&json_len, 8, 1, fp);
    fwrite(json, 1, (size_t)json_len, fp);
    for (i = 0; i < n_tensors; ++i)
        fwrite(dummy, 1, 256, fp);
    fclose(fp);
    return 0;
}

static void fwrite_u32(FILE *fp, uint32_t v) { fwrite(&v, 4, 1, fp); }
static void fwrite_u64(FILE *fp, uint64_t v) { fwrite(&v, 8, 1, fp); }

static int gen_gguf(const char *path, int n_tensors) {
    FILE *fp;
    long pos_l, aligned;
    int i;
    float dummy[64];

    memset(dummy, 0, sizeof(dummy));
    fp = fopen(path, "wb");
    if (!fp) return -1;

    fwrite_u32(fp, 0x46554747u);
    fwrite_u32(fp, 3);
    fwrite_u64(fp, (uint64_t)n_tensors);
    fwrite_u64(fp, 0);

    for (i = 0; i < n_tensors; ++i) {
        char name[16];
        uint64_t nlen;
        snprintf(name, sizeof(name), "t_%04d", i);
        nlen = (uint64_t)strlen(name);
        fwrite_u64(fp, nlen);
        fwrite(name, 1, (size_t)nlen, fp);
        fwrite_u32(fp, 2);  /* n_dims */
        fwrite_u64(fp, 8);
        fwrite_u64(fp, 8);
        fwrite_u32(fp, 0);  /* F32 */
        fwrite_u64(fp, (uint64_t)i * 256);
    }

    pos_l = ftell(fp);
    aligned = (pos_l + 63) & ~63L;
    while (pos_l < aligned) { uint8_t z = 0; fwrite(&z, 1, 1, fp); pos_l++; }

    for (i = 0; i < n_tensors; ++i)
        fwrite(dummy, 1, 256, fp);

    fclose(fp);
    return 0;
}

static int gen_tq(const char *path, int n_tensors) {
    FILE *fp;
    tq_header_t hdr;
    long pos_l, aligned;
    int i;
    uint8_t dummy[4]; /* b=2, 4x4 = 16 vals, 4 bytes */

    memset(dummy, 0, sizeof(dummy));
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = 0x46555154u;
    hdr.version = 2;
    hdr.tensor_count = (uint64_t)n_tensors;
    hdr.model_family_id = 0;

    pos_l = (long)(sizeof(tq_header_t) + (size_t)n_tensors * sizeof(tq_tensor_t));
    aligned = (pos_l + 63) & ~63L;
    hdr.data_offset = (uint64_t)aligned;
    hdr.total_data_size = (uint64_t)n_tensors * 4;

    fp = fopen(path, "wb");
    if (!fp) return -1;
    fwrite(&hdr, sizeof(hdr), 1, fp);

    for (i = 0; i < n_tensors; ++i) {
        tq_tensor_t desc;
        memset(&desc, 0, sizeof(desc));
        snprintf(desc.name, sizeof(desc.name), "t_%04d", i);
        desc.b = 2;
        desc.rows = 4;
        desc.cols = 4;
        desc.frame_offset = (uint64_t)i * 4;
        desc.unpacked_size = 4;
        fwrite(&desc, sizeof(desc), 1, fp);
    }

    pos_l = ftell(fp);
    while (pos_l < aligned) { uint8_t z = 0; fwrite(&z, 1, 1, fp); pos_l++; }

    for (i = 0; i < n_tensors; ++i)
        fwrite(dummy, 1, 4, fp);

    fclose(fp);
    return 0;
}

/* ----------------------------------------------------------------
 * Benchmark runners
 * ---------------------------------------------------------------- */

static void bench_safetensors(const char *path) {
    struct timespec t0, t1;
    double total = 0;
    int i;

    for (i = 0; i < WARMUP_ITERS; ++i) {
        st_mmap_t mm; st_file_t f;
        st_mmap(path, &mm); st_parse(&mm, &f); st_free(&f); st_munmap(&mm);
    }

    for (i = 0; i < BENCH_ITERS; ++i) {
        st_mmap_t mm; st_file_t f;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        st_mmap(path, &mm);
        st_parse(&mm, &f);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        total += elapsed_ms(&t0, &t1);
        st_free(&f); st_munmap(&mm);
    }
    printf("  Safetensors  mmap+parse:  %.3f ms  (avg of %d)\n",
           total / BENCH_ITERS, BENCH_ITERS);
}

static void bench_gguf(const char *path) {
    struct timespec t0, t1;
    double total = 0;
    int i;

    for (i = 0; i < WARMUP_ITERS; ++i) {
        gguf_mmap_t mm; gguf_file_t f;
        gguf_mmap(path, &mm); gguf_parse(&mm, &f); gguf_free(&f); gguf_munmap(&mm);
    }

    for (i = 0; i < BENCH_ITERS; ++i) {
        gguf_mmap_t mm; gguf_file_t f;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        gguf_mmap(path, &mm);
        gguf_parse(&mm, &f);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        total += elapsed_ms(&t0, &t1);
        gguf_free(&f); gguf_munmap(&mm);
    }
    printf("  GGUF         mmap+parse:  %.3f ms  (avg of %d)\n",
           total / BENCH_ITERS, BENCH_ITERS);
}

static void bench_tq(const char *path) {
    struct timespec t0, t1;
    double total = 0;
    int i;

    for (i = 0; i < WARMUP_ITERS; ++i) {
        tq_file_t f;
        tq_mmap(path, &f); tq_munmap(&f);
    }

    for (i = 0; i < BENCH_ITERS; ++i) {
        tq_file_t f;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        tq_mmap(path, &f);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        total += elapsed_ms(&t0, &t1);
        tq_munmap(&f);
    }
    printf("  TQ           mmap+parse:  %.3f ms  (avg of %d)\n",
           total / BENCH_ITERS, BENCH_ITERS);
}

/* ----------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------- */

int main(int argc, char **argv) {
    int n = 100;
    const char *st_path = "/tmp/tensio_bench.safetensors";
    const char *gguf_path = "/tmp/tensio_bench.gguf";
    const char *tq_path = "/tmp/tensio_bench.tq";

    if (argc > 1) n = atoi(argv[1]);
    if (n < 1) n = 100;

    printf("=== Parse Benchmark (%d tensors) ===\n", n);

    gen_safetensors(st_path, n);
    gen_gguf(gguf_path, n);
    gen_tq(tq_path, n);

    bench_safetensors(st_path);
    bench_gguf(gguf_path);
    bench_tq(tq_path);

    remove(st_path);
    remove(gguf_path);
    remove(tq_path);
    return 0;
}
