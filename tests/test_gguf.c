/* test_gguf.c — tests for GGUF mmap, parse, metadata, write, round-trip */

#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-50s ", #name); \
    if (test_##name()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

/* ----------------------------------------------------------------
 * Helper: create a minimal GGUF v3 file on disk
 *
 * Layout:
 *   [4] magic "GGUF"
 *   [4] version = 3
 *   [8] tensor_count = 1
 *   [8] metadata_kv_count = 1
 *   --- metadata KV ---
 *   [8] key_len + key bytes ("general.name")
 *   [4] value_type = GGUF_META_STRING (8)
 *   [8] str_len + str bytes ("test-model")
 *   --- tensor info ---
 *   [8] name_len + name bytes ("weight")
 *   [4] n_dims = 2
 *   [8] ne[0] = 4
 *   [8] ne[1] = 3
 *   [4] type = GGUF_TYPE_F32 (0)
 *   [8] offset = 0
 *   --- 64-byte aligned padding ---
 *   --- tensor data: 4*3*4 = 48 bytes of F32 ---
 * ---------------------------------------------------------------- */

static const char *FIXTURE_PATH = "/tmp/tensio_test.gguf";

static void write_u32(FILE *fp, uint32_t v) { fwrite(&v, 4, 1, fp); }
static void write_u64(FILE *fp, uint64_t v) { fwrite(&v, 8, 1, fp); }
static void write_str(FILE *fp, const char *s) {
    uint64_t len = (uint64_t)strlen(s);
    write_u64(fp, len);
    fwrite(s, 1, (size_t)len, fp);
}

static int write_fixture(void) {
    FILE *fp;
    float data[12];
    long pos, aligned;
    int i;

    fp = fopen(FIXTURE_PATH, "wb");
    if (!fp) return -1;

    /* Header */
    write_u32(fp, 0x46554747u);  /* GGUF magic */
    write_u32(fp, 3);            /* version */
    write_u64(fp, 1);            /* tensor_count */
    write_u64(fp, 1);            /* metadata_kv_count */

    /* Metadata: "general.name" = "test-model" */
    write_str(fp, "general.name");
    write_u32(fp, 8);  /* GGUF_META_STRING */
    write_str(fp, "test-model");

    /* Tensor info: "weight" F32 [4,3] */
    write_str(fp, "weight");
    write_u32(fp, 2);  /* n_dims */
    write_u64(fp, 4);  /* ne[0] */
    write_u64(fp, 3);  /* ne[1] */
    write_u32(fp, 0);  /* GGUF_TYPE_F32 */
    write_u64(fp, 0);  /* offset */

    /* Pad to 64-byte alignment */
    pos = ftell(fp);
    aligned = (pos + 63) & ~63L;
    while (pos < aligned) {
        uint8_t zero = 0;
        fwrite(&zero, 1, 1, fp);
        pos++;
    }

    /* Tensor data */
    for (i = 0; i < 12; ++i) data[i] = (float)(i + 1);
    fwrite(data, sizeof(float), 12, fp);

    fclose(fp);
    return 0;
}

/* ----------------------------------------------------------------
 * Tests
 * ---------------------------------------------------------------- */

static int test_mmap_valid(void) {
    gguf_mmap_t mm;
    if (gguf_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (mm.base == NULL || mm.size == 0) { gguf_munmap(&mm); return 0; }
    gguf_munmap(&mm);
    return 1;
}

static int test_mmap_nonexistent(void) {
    gguf_mmap_t mm;
    return gguf_mmap("/tmp/does_not_exist_tensio.gguf", &mm) != 0;
}

static int test_parse_header(void) {
    gguf_mmap_t mm;
    gguf_file_t f;
    int ok;
    if (gguf_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (gguf_parse(&mm, &f) != 0) { gguf_munmap(&mm); return 0; }
    ok = (f.magic == 0x46554747u && f.version == 3 &&
          f.tensor_count == 1 && f.metadata_count == 1);
    gguf_free(&f);
    gguf_munmap(&mm);
    return ok;
}

static int test_parse_metadata(void) {
    gguf_mmap_t mm;
    gguf_file_t f;
    const gguf_kv_t *kv;
    int ok;
    if (gguf_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (gguf_parse(&mm, &f) != 0) { gguf_munmap(&mm); return 0; }
    kv = gguf_get_kv(&f, "general.name");
    ok = (kv != NULL && kv->type == GGUF_META_STRING &&
          kv->value.str.data != NULL &&
          strcmp(kv->value.str.data, "test-model") == 0);
    gguf_free(&f);
    gguf_munmap(&mm);
    return ok;
}

static int test_parse_tensor_info(void) {
    gguf_mmap_t mm;
    gguf_file_t f;
    const gguf_tensor_t *t;
    int ok;
    if (gguf_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (gguf_parse(&mm, &f) != 0) { gguf_munmap(&mm); return 0; }
    t = gguf_get_tensor(&f, "weight");
    ok = (t != NULL && t->type == GGUF_TYPE_F32 &&
          t->n_dims == 2 && t->ne[0] == 4 && t->ne[1] == 3);
    gguf_free(&f);
    gguf_munmap(&mm);
    return ok;
}

static int test_tensor_data(void) {
    gguf_mmap_t mm;
    gguf_file_t f;
    const gguf_tensor_t *t;
    float *data;
    int ok;
    if (gguf_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (gguf_parse(&mm, &f) != 0) { gguf_munmap(&mm); return 0; }
    t = gguf_get_tensor(&f, "weight");
    if (!t) { gguf_free(&f); gguf_munmap(&mm); return 0; }
    data = (float *)gguf_get_tensor_data(&f, t);
    ok = (data[0] == 1.0f && data[11] == 12.0f);
    gguf_free(&f);
    gguf_munmap(&mm);
    return ok;
}

static int test_tensor_size(void) {
    gguf_mmap_t mm;
    gguf_file_t f;
    const gguf_tensor_t *t;
    int ok;
    if (gguf_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (gguf_parse(&mm, &f) != 0) { gguf_munmap(&mm); return 0; }
    t = gguf_get_tensor(&f, "weight");
    /* 4*3 F32 = 48 bytes */
    ok = (t != NULL && t->size == 48);
    gguf_free(&f);
    gguf_munmap(&mm);
    return ok;
}

static int test_get_tensor_missing(void) {
    gguf_mmap_t mm;
    gguf_file_t f;
    const gguf_tensor_t *t;
    if (gguf_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (gguf_parse(&mm, &f) != 0) { gguf_munmap(&mm); return 0; }
    t = gguf_get_tensor(&f, "nonexistent");
    gguf_free(&f);
    gguf_munmap(&mm);
    return (t == NULL);
}

static int test_get_kv_missing(void) {
    gguf_mmap_t mm;
    gguf_file_t f;
    const gguf_kv_t *kv;
    if (gguf_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (gguf_parse(&mm, &f) != 0) { gguf_munmap(&mm); return 0; }
    kv = gguf_get_kv(&f, "nonexistent.key");
    gguf_free(&f);
    gguf_munmap(&mm);
    return (kv == NULL);
}

static int test_write_roundtrip(void) {
    const char *out_path = "/tmp/tensio_test_rt.gguf";
    gguf_mmap_t mm1, mm2;
    gguf_file_t f1, f2;
    const gguf_tensor_t *t1, *t2;
    float *d1, *d2;
    int ok = 0;

    if (gguf_mmap(FIXTURE_PATH, &mm1) != 0) return 0;
    if (gguf_parse(&mm1, &f1) != 0) { gguf_munmap(&mm1); return 0; }

    if (gguf_write(out_path, &f1) != 0) {
        gguf_free(&f1); gguf_munmap(&mm1); return 0;
    }

    if (gguf_mmap(out_path, &mm2) != 0) {
        gguf_free(&f1); gguf_munmap(&mm1); return 0;
    }
    if (gguf_parse(&mm2, &f2) != 0) {
        gguf_munmap(&mm2); gguf_free(&f1); gguf_munmap(&mm1); return 0;
    }

    if (f1.tensor_count == f2.tensor_count &&
        f1.metadata_count == f2.metadata_count) {
        t1 = gguf_get_tensor(&f1, "weight");
        t2 = gguf_get_tensor(&f2, "weight");
        if (t1 && t2 && t1->size == t2->size) {
            d1 = (float *)gguf_get_tensor_data(&f1, t1);
            d2 = (float *)gguf_get_tensor_data(&f2, t2);
            ok = (memcmp(d1, d2, (size_t)t1->size) == 0);
        }
    }

    gguf_free(&f2); gguf_munmap(&mm2);
    gguf_free(&f1); gguf_munmap(&mm1);
    remove(out_path);
    return ok;
}

static int test_parse_bad_magic(void) {
    const char *path = "/tmp/tensio_test_bad.gguf";
    gguf_mmap_t mm;
    gguf_file_t f;
    int ok;
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    write_u32(fp, 0xDEADBEEF);  /* wrong magic */
    write_u32(fp, 3);
    write_u64(fp, 0);
    write_u64(fp, 0);
    fclose(fp);

    if (gguf_mmap(path, &mm) != 0) { remove(path); return 0; }
    ok = (gguf_parse(&mm, &f) != 0);
    gguf_munmap(&mm);
    remove(path);
    return ok;
}

/* ----------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------- */

int main(void) {
    printf("=== GGUF Tests ===\n");

    if (write_fixture() != 0) {
        fprintf(stderr, "Failed to write test fixture\n");
        return 1;
    }

    TEST(mmap_valid);
    TEST(mmap_nonexistent);
    TEST(parse_header);
    TEST(parse_metadata);
    TEST(parse_tensor_info);
    TEST(tensor_data);
    TEST(tensor_size);
    TEST(get_tensor_missing);
    TEST(get_kv_missing);
    TEST(write_roundtrip);
    TEST(parse_bad_magic);

    remove(FIXTURE_PATH);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
