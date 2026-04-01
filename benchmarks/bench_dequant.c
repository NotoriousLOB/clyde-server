#define _POSIX_C_SOURCE 199309L
/* bench_dequant.c — TQ dequantization throughput benchmark
 *
 * Measures throughput of tq_dequant for b=2 and b=3 ternary formats.
 *
 * Usage: bench_dequant [rows] [cols]    (default: 4096 4096)
 */

#include "tq.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WARMUP_ITERS 2
#define BENCH_ITERS  5

static double elapsed_ms(struct timespec *start, struct timespec *end) {
    double s = (double)(end->tv_sec - start->tv_sec) * 1e3;
    double ns = (double)(end->tv_nsec - start->tv_nsec) / 1e6;
    return s + ns;
}

static int gen_tq_file(const char *path, uint32_t rows, uint32_t cols,
                       uint32_t b) {
    FILE *fp;
    tq_header_t hdr;
    tq_tensor_t desc;
    long pos, aligned;
    uint64_t packed_size;
    uint8_t *data;

    if (b == 2) {
        packed_size = ((uint64_t)rows * cols + 3) / 4;
    } else {
        /* b=3: 3 bits per value, packed */
        packed_size = ((uint64_t)rows * cols * 3 + 7) / 8;
    }

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = 0x46555154u;
    hdr.version = 2;
    hdr.tensor_count = 1;

    memset(&desc, 0, sizeof(desc));
    snprintf(desc.name, sizeof(desc.name), "weight");
    desc.b = b;
    desc.rows = rows;
    desc.cols = cols;
    desc.frame_offset = 0;
    desc.unpacked_size = packed_size;

    pos = (long)(sizeof(hdr) + sizeof(desc));
    aligned = (pos + 63) & ~63L;
    hdr.data_offset = (uint64_t)aligned;
    hdr.total_data_size = packed_size;

    fp = fopen(path, "wb");
    if (!fp) return -1;
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fwrite(&desc, sizeof(desc), 1, fp);

    pos = ftell(fp);
    while (pos < aligned) { uint8_t z = 0; fwrite(&z, 1, 1, fp); pos++; }

    /* Write random-ish packed data */
    data = (uint8_t *)calloc(1, (size_t)packed_size);
    if (data) {
        size_t j;
        for (j = 0; j < (size_t)packed_size; ++j)
            data[j] = (uint8_t)(j * 37 + 13);
        fwrite(data, 1, (size_t)packed_size, fp);
        free(data);
    }

    fclose(fp);
    return 0;
}

static void bench_dequant(const char *label, const char *path,
                          uint32_t rows, uint32_t cols) {
    struct timespec t0, t1;
    double total = 0;
    uint64_t n_elements = (uint64_t)rows * cols;
    float *dst;
    int i;
    tq_file_t f;

    dst = (float *)malloc(n_elements * sizeof(float));
    if (!dst) { fprintf(stderr, "OOM\n"); return; }

    if (tq_mmap(path, &f) != 0) {
        fprintf(stderr, "Failed to mmap %s\n", path);
        free(dst);
        return;
    }

    for (i = 0; i < WARMUP_ITERS; ++i)
        tq_dequant(&f, 0, dst);

    for (i = 0; i < BENCH_ITERS; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        tq_dequant(&f, 0, dst);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        total += elapsed_ms(&t0, &t1);
    }

    {
        double avg_ms = total / BENCH_ITERS;
        double bytes = (double)n_elements * 4.0; /* output F32 */
        double gb_per_s = (bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_ms / 1000.0);
        printf("  %-12s  %ux%u  %.3f ms  (%.2f GB/s output)\n",
               label, rows, cols, avg_ms, gb_per_s);
    }

    tq_munmap(&f);
    free(dst);
}

int main(int argc, char **argv) {
    uint32_t rows = 4096, cols = 4096;
    const char *path_b2 = "/tmp/tensio_bench_b2.tq";
    const char *path_b3 = "/tmp/tensio_bench_b3.tq";

    if (argc > 1) rows = (uint32_t)atoi(argv[1]);
    if (argc > 2) cols = (uint32_t)atoi(argv[2]);
    if (rows == 0) rows = 4096;
    if (cols == 0) cols = 4096;

    printf("=== Dequant Benchmark ===\n");

    gen_tq_file(path_b2, rows, cols, 2);
    gen_tq_file(path_b3, rows, cols, 3);

    bench_dequant("b=2 ternary", path_b2, rows, cols);
    bench_dequant("b=3 ternary", path_b3, rows, cols);

    remove(path_b2);
    remove(path_b3);
    return 0;
}
