#define _POSIX_C_SOURCE 199309L
/* bench_convert.c — Conversion benchmark (identity path only for now)
 *
 * Measures convert_any_to_any latency for identity (same-format) copies
 * of each supported format.
 *
 * Usage: bench_convert [num_tensors]    (default: 100)
 */

#include "safetensors.h"
#include "gguf.h"
#include "tq.h"
#include "convert.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BENCH_ITERS 5

static double elapsed_ms(struct timespec *start, struct timespec *end) {
    double s = (double)(end->tv_sec - start->tv_sec) * 1e3;
    double ns = (double)(end->tv_nsec - start->tv_nsec) / 1e6;
    return s + ns;
}

/* ----------------------------------------------------------------
 * Fixture generators (same as bench_parse — minimal versions)
 * ---------------------------------------------------------------- */

static void fwrite_u32(FILE *fp, uint32_t v) { fwrite(&v, 4, 1, fp); }
static void fwrite_u64(FILE *fp, uint64_t v) { fwrite(&v, 8, 1, fp); }

static int gen_safetensors(const char *path, int n) {
    FILE *fp;
    char json[1024 * 64];
    int pos = 0;
    uint64_t data_off = 0, json_len;
    float dummy[64];
    int i;

    memset(dummy, 0, sizeof(dummy));
    pos += snprintf(json + pos, sizeof(json) - (size_t)pos, "{");
    for (i = 0; i < n; ++i) {
        uint64_t sz = 256;
        if (i > 0) pos += snprintf(json + pos, sizeof(json) - (size_t)pos, ",");
        pos += snprintf(json + pos, sizeof(json) - (size_t)pos,
            "\"t_%04d\":{\"dtype\":\"F32\",\"shape\":[8,8],"
            "\"data_offsets\":[%llu,%llu]}", i,
            (unsigned long long)data_off,
            (unsigned long long)(data_off + sz));
        data_off += sz;
    }
    pos += snprintf(json + pos, sizeof(json) - (size_t)pos, "}");
    json_len = (uint64_t)pos;

    fp = fopen(path, "wb");
    if (!fp) return -1;
    fwrite(&json_len, 8, 1, fp);
    fwrite(json, 1, (size_t)json_len, fp);
    for (i = 0; i < n; ++i) fwrite(dummy, 1, 256, fp);
    fclose(fp);
    return 0;
}

static int gen_gguf(const char *path, int n) {
    FILE *fp;
    long pos_l, aligned;
    float dummy[64];
    int i;

    memset(dummy, 0, sizeof(dummy));
    fp = fopen(path, "wb");
    if (!fp) return -1;
    fwrite_u32(fp, 0x46554747u);
    fwrite_u32(fp, 3);
    fwrite_u64(fp, (uint64_t)n);
    fwrite_u64(fp, 0);
    for (i = 0; i < n; ++i) {
        char name[16];
        uint64_t nlen;
        snprintf(name, sizeof(name), "t_%04d", i);
        nlen = (uint64_t)strlen(name);
        fwrite_u64(fp, nlen);
        fwrite(name, 1, (size_t)nlen, fp);
        fwrite_u32(fp, 2);
        fwrite_u64(fp, 8); fwrite_u64(fp, 8);
        fwrite_u32(fp, 0);
        fwrite_u64(fp, (uint64_t)i * 256);
    }
    pos_l = ftell(fp);
    aligned = (pos_l + 63) & ~63L;
    while (pos_l < aligned) { uint8_t z = 0; fwrite(&z, 1, 1, fp); pos_l++; }
    for (i = 0; i < n; ++i) fwrite(dummy, 1, 256, fp);
    fclose(fp);
    return 0;
}

static int gen_tq(const char *path, int n) {
    FILE *fp;
    tq_header_t hdr;
    long pos_l, aligned;
    uint8_t dummy[4];
    int i;

    memset(dummy, 0, sizeof(dummy));
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = 0x46555154u;
    hdr.version = 2;
    hdr.tensor_count = (uint64_t)n;
    pos_l = (long)(sizeof(hdr) + (size_t)n * sizeof(tq_tensor_t));
    aligned = (pos_l + 63) & ~63L;
    hdr.data_offset = (uint64_t)aligned;
    hdr.total_data_size = (uint64_t)n * 4;

    fp = fopen(path, "wb");
    if (!fp) return -1;
    fwrite(&hdr, sizeof(hdr), 1, fp);
    for (i = 0; i < n; ++i) {
        tq_tensor_t desc;
        memset(&desc, 0, sizeof(desc));
        snprintf(desc.name, sizeof(desc.name), "t_%04d", i);
        desc.b = 2; desc.rows = 4; desc.cols = 4;
        desc.frame_offset = (uint64_t)i * 4;
        desc.unpacked_size = 4;
        fwrite(&desc, sizeof(desc), 1, fp);
    }
    pos_l = ftell(fp);
    while (pos_l < aligned) { uint8_t z = 0; fwrite(&z, 1, 1, fp); pos_l++; }
    for (i = 0; i < n; ++i) fwrite(dummy, 1, 4, fp);
    fclose(fp);
    return 0;
}

/* ----------------------------------------------------------------
 * Benchmark runner
 * ---------------------------------------------------------------- */

static void bench_identity(const char *label, const char *in,
                           const char *out) {
    struct timespec t0, t1;
    double total = 0;
    int i;

    for (i = 0; i < BENCH_ITERS; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        convert_any_to_any(in, out);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        total += elapsed_ms(&t0, &t1);
        remove(out);
    }
    printf("  %-14s identity copy:  %.3f ms  (avg of %d)\n",
           label, total / BENCH_ITERS, BENCH_ITERS);
}

int main(int argc, char **argv) {
    int n = 100;
    const char *st_in  = "/tmp/tensio_bench_conv.safetensors";
    const char *gguf_in = "/tmp/tensio_bench_conv.gguf";
    const char *tq_in  = "/tmp/tensio_bench_conv.tq";

    if (argc > 1) n = atoi(argv[1]);
    if (n < 1) n = 100;

    printf("=== Convert Benchmark (%d tensors) ===\n", n);

    gen_safetensors(st_in, n);
    gen_gguf(gguf_in, n);
    gen_tq(tq_in, n);

    bench_identity("Safetensors", st_in, "/tmp/tensio_bench_out.safetensors");
    bench_identity("GGUF", gguf_in, "/tmp/tensio_bench_out.gguf");
    bench_identity("TQ", tq_in, "/tmp/tensio_bench_out.tq");

    remove(st_in);
    remove(gguf_in);
    remove(tq_in);
    return 0;
}
