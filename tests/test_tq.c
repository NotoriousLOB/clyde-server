/* test_tq.c — tests for TQ format mmap, parse, dequant, write, round-trip */

#include "tq.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-50s ", #name); \
    if (test_##name()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

/* ----------------------------------------------------------------
 * Helper: create a minimal TQ v2 file on disk
 *
 * Layout:
 *   [48] tq_header_t
 *   [192] tq_tensor_t descriptor for "weight" (b=2, 4x4)
 *   [padding to 64-byte alignment]
 *   [data: 2-bit packed ternary, 4*4 / 4 = 4 bytes]
 * ---------------------------------------------------------------- */

static const char *FIXTURE_PATH = "/tmp/tensio_test.tq";

static int write_fixture(void) {
    FILE *fp;
    tq_header_t hdr;
    tq_tensor_t desc;
    long pos, aligned;
    /* 4x4 ternary b=2: 16 values, 4 per byte = 4 bytes of packed data */
    /* encode: 0=-1, 1=0, 2=+1 */
    /* row 0: [-1, 0, +1, -1] => vals [0,1,2,0] => byte: 0 | (1<<2) | (2<<4) | (0<<6) = 0x24 */
    uint8_t packed[4] = {0x24, 0x24, 0x24, 0x24};

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = TQ_MAGIC;
    hdr.version = TQ_VERSION;
    hdr.features = 0;
    hdr.tensor_count = 1;
    hdr.model_family_id = TQ_FAMILY_UNKNOWN;
    hdr.model_version = 0;

    memset(&desc, 0, sizeof(desc));
    strncpy(desc.name, "weight", sizeof(desc.name) - 1);
    desc.b = 2;
    desc.rows = 4;
    desc.cols = 4;
    desc.tensor_flags = 0;
    desc.wht_seed = 0;
    desc.frame_offset = 0;
    desc.frame_size = 0; /* uncompressed */
    desc.unpacked_size = 4;
    desc.index_size = 0;
    desc.norm_offset = 0;

    /* Compute data_offset: header + 1 descriptor, aligned to 64 */
    pos = (long)(sizeof(tq_header_t) + sizeof(tq_tensor_t));
    aligned = (pos + 63) & ~63L;
    hdr.data_offset = (uint64_t)aligned;
    hdr.total_data_size = 4;

    fp = fopen(FIXTURE_PATH, "wb");
    if (!fp) return -1;

    fwrite(&hdr, sizeof(hdr), 1, fp);
    fwrite(&desc, sizeof(desc), 1, fp);

    /* Pad */
    pos = ftell(fp);
    while (pos < aligned) {
        uint8_t zero = 0;
        fwrite(&zero, 1, 1, fp);
        pos++;
    }

    fwrite(packed, 1, 4, fp);
    fclose(fp);
    return 0;
}

/* ----------------------------------------------------------------
 * Tests
 * ---------------------------------------------------------------- */

static int test_mmap_valid(void) {
    tq_file_t f;
    if (tq_mmap(FIXTURE_PATH, &f) != 0) return 0;
    if (f.base == NULL || f.size == 0) { tq_munmap(&f); return 0; }
    tq_munmap(&f);
    return 1;
}

static int test_mmap_nonexistent(void) {
    tq_file_t f;
    return tq_mmap("/tmp/does_not_exist_tensio.tq", &f) != 0;
}

static int test_parse_header(void) {
    tq_file_t f;
    int ok;
    if (tq_mmap(FIXTURE_PATH, &f) != 0) return 0;
    ok = (f.hdr->magic == TQ_MAGIC && f.hdr->version == TQ_VERSION &&
          f.hdr->tensor_count == 1);
    tq_munmap(&f);
    return ok;
}

static int test_parse_tensor_desc(void) {
    tq_file_t f;
    int ok;
    if (tq_mmap(FIXTURE_PATH, &f) != 0) return 0;
    ok = (strcmp(f.tensors[0].name, "weight") == 0 &&
          f.tensors[0].b == 2 &&
          f.tensors[0].rows == 4 &&
          f.tensors[0].cols == 4);
    tq_munmap(&f);
    return ok;
}

static int test_get_tensor_data(void) {
    tq_file_t f;
    uint8_t *data;
    int ok;
    if (tq_mmap(FIXTURE_PATH, &f) != 0) return 0;
    data = (uint8_t *)tq_get_tensor_data(&f, &f.tensors[0]);
    ok = (data != NULL && data[0] == 0x24);
    tq_munmap(&f);
    return ok;
}

static int test_dequant_b2(void) {
    tq_file_t f;
    float dst[16];
    int ok;
    if (tq_mmap(FIXTURE_PATH, &f) != 0) return 0;
    memset(dst, 0, sizeof(dst));
    tq_dequant(&f, 0, dst);
    /* byte 0x24 = 00 10 01 00 => vals [0,1,2,0] => [-1.0, -0.33, 0.33, -1.0] */
    ok = (dst[0] == -1.0f && dst[1] > -0.35f && dst[1] < -0.30f &&
          dst[2] > 0.30f && dst[2] < 0.35f && dst[3] == -1.0f);
    tq_munmap(&f);
    return ok;
}

static int test_write_roundtrip(void) {
    const char *out_path = "/tmp/tensio_test_rt.tq";
    tq_file_t f1, f2;
    uint8_t *d1, *d2;
    int ok = 0;

    if (tq_mmap(FIXTURE_PATH, &f1) != 0) return 0;

    if (tq_write(out_path, &f1) != 0) {
        tq_munmap(&f1); return 0;
    }

    if (tq_mmap(out_path, &f2) != 0) {
        tq_munmap(&f1); return 0;
    }

    if (f1.hdr->tensor_count == f2.hdr->tensor_count) {
        d1 = (uint8_t *)tq_get_tensor_data(&f1, &f1.tensors[0]);
        d2 = (uint8_t *)tq_get_tensor_data(&f2, &f2.tensors[0]);
        ok = (memcmp(d1, d2, (size_t)f1.tensors[0].unpacked_size) == 0);
    }

    tq_munmap(&f2);
    tq_munmap(&f1);
    remove(out_path);
    return ok;
}

static int test_mmap_bad_magic(void) {
    const char *path = "/tmp/tensio_test_bad.tq";
    tq_file_t f;
    int ok;
    tq_header_t hdr;
    FILE *fp;

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = 0xDEADBEEF;
    hdr.version = TQ_VERSION;

    fp = fopen(path, "wb");
    if (!fp) return 0;
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fclose(fp);

    ok = (tq_mmap(path, &f) != 0);
    remove(path);
    return ok;
}

static int test_model_family(void) {
    tq_file_t f;
    int ok;
    if (tq_mmap(FIXTURE_PATH, &f) != 0) return 0;
    ok = (f.hdr->model_family_id == TQ_FAMILY_UNKNOWN);
    tq_munmap(&f);
    return ok;
}

/* ----------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------- */

int main(void) {
    printf("=== TQ Tests ===\n");

    if (write_fixture() != 0) {
        fprintf(stderr, "Failed to write test fixture\n");
        return 1;
    }

    TEST(mmap_valid);
    TEST(mmap_nonexistent);
    TEST(parse_header);
    TEST(parse_tensor_desc);
    TEST(get_tensor_data);
    TEST(dequant_b2);
    TEST(write_roundtrip);
    TEST(mmap_bad_magic);
    TEST(model_family);

    remove(FIXTURE_PATH);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
