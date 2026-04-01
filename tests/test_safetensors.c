/* test_safetensors.c — tests for Safetensors mmap, parse, write, round-trip */

#include "safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-50s ", #name); \
    if (test_##name()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

/* ----------------------------------------------------------------
 * Helper: create a minimal safetensors file on disk
 * Format: 8-byte header_len (LE) + JSON header + raw data
 * ---------------------------------------------------------------- */

static const char *FIXTURE_PATH = "/tmp/tensio_test.safetensors";

static int write_fixture(void) {
    FILE *fp;
    const char *json =
        "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2,3],"
        "\"data_offsets\":[0,24]},"
        "\"bias\":{\"dtype\":\"F32\",\"shape\":[3],"
        "\"data_offsets\":[24,36]}}";
    uint64_t json_len = (uint64_t)strlen(json);
    /* 2x3 floats = 24 bytes, 3 floats = 12 bytes, total 36 bytes */
    float data[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                     0.1f, 0.2f, 0.3f};

    fp = fopen(FIXTURE_PATH, "wb");
    if (!fp) return -1;
    fwrite(&json_len, 8, 1, fp);
    fwrite(json, 1, (size_t)json_len, fp);
    fwrite(data, sizeof(float), 9, fp);
    fclose(fp);
    return 0;
}

/* ----------------------------------------------------------------
 * Tests
 * ---------------------------------------------------------------- */

static int test_mmap_valid_file(void) {
    st_mmap_t mm;
    if (st_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (mm.base == NULL || mm.size == 0) { st_munmap(&mm); return 0; }
    st_munmap(&mm);
    return 1;
}

static int test_mmap_nonexistent(void) {
    st_mmap_t mm;
    return st_mmap("/tmp/does_not_exist_tensio.safetensors", &mm) != 0;
}

static int test_parse_tensor_count(void) {
    st_mmap_t mm;
    st_file_t f;
    int ok;
    if (st_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (st_parse(&mm, &f) != 0) { st_munmap(&mm); return 0; }
    ok = (f.num_tensors == 2);
    st_free(&f);
    st_munmap(&mm);
    return ok;
}

static int test_parse_tensor_names(void) {
    st_mmap_t mm;
    st_file_t f;
    int found_weight = 0, found_bias = 0;
    uint32_t i;
    if (st_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (st_parse(&mm, &f) != 0) { st_munmap(&mm); return 0; }
    for (i = 0; i < f.num_tensors; ++i) {
        if (strcmp(f.tensors[i].name, "weight") == 0) found_weight = 1;
        if (strcmp(f.tensors[i].name, "bias") == 0) found_bias = 1;
    }
    st_free(&f);
    st_munmap(&mm);
    return found_weight && found_bias;
}

static int test_parse_dtype(void) {
    st_mmap_t mm;
    st_file_t f;
    const st_tensor_t *t;
    int ok;
    if (st_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (st_parse(&mm, &f) != 0) { st_munmap(&mm); return 0; }
    t = st_get_tensor(&f, "weight");
    ok = (t != NULL && t->dtype == ST_F32);
    st_free(&f);
    st_munmap(&mm);
    return ok;
}

static int test_parse_shape(void) {
    st_mmap_t mm;
    st_file_t f;
    const st_tensor_t *t;
    int ok;
    if (st_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (st_parse(&mm, &f) != 0) { st_munmap(&mm); return 0; }
    t = st_get_tensor(&f, "weight");
    ok = (t != NULL && t->ndim == 2 && t->shape[0] == 2 && t->shape[1] == 3);
    st_free(&f);
    st_munmap(&mm);
    return ok;
}

static int test_parse_data_offsets(void) {
    st_mmap_t mm;
    st_file_t f;
    const st_tensor_t *tw, *tb;
    int ok;
    if (st_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (st_parse(&mm, &f) != 0) { st_munmap(&mm); return 0; }
    tw = st_get_tensor(&f, "weight");
    tb = st_get_tensor(&f, "bias");
    ok = (tw && tw->offset == 0 && tw->size == 24 &&
           tb && tb->offset == 24 && tb->size == 12);
    st_free(&f);
    st_munmap(&mm);
    return ok;
}

static int test_get_tensor_data(void) {
    st_mmap_t mm;
    st_file_t f;
    const st_tensor_t *t;
    float *data;
    int ok;
    if (st_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (st_parse(&mm, &f) != 0) { st_munmap(&mm); return 0; }
    t = st_get_tensor(&f, "weight");
    data = (float *)st_get_tensor_data(&f, t);
    ok = (data != NULL && data[0] == 1.0f && data[5] == 6.0f);
    st_free(&f);
    st_munmap(&mm);
    return ok;
}

static int test_get_tensor_missing(void) {
    st_mmap_t mm;
    st_file_t f;
    const st_tensor_t *t;
    if (st_mmap(FIXTURE_PATH, &mm) != 0) return 0;
    if (st_parse(&mm, &f) != 0) { st_munmap(&mm); return 0; }
    t = st_get_tensor(&f, "nonexistent");
    st_free(&f);
    st_munmap(&mm);
    return (t == NULL);
}

static int test_write_roundtrip(void) {
    const char *out_path = "/tmp/tensio_test_rt.safetensors";
    st_mmap_t mm1, mm2;
    st_file_t f1, f2;
    const st_tensor_t *tw1, *tw2;
    float *d1, *d2;
    int ok = 0;

    if (st_mmap(FIXTURE_PATH, &mm1) != 0) return 0;
    if (st_parse(&mm1, &f1) != 0) { st_munmap(&mm1); return 0; }

    /* Write out */
    if (st_write(out_path, &f1) != 0) {
        st_free(&f1); st_munmap(&mm1); return 0;
    }

    /* Read back */
    if (st_mmap(out_path, &mm2) != 0) {
        st_free(&f1); st_munmap(&mm1); return 0;
    }
    if (st_parse(&mm2, &f2) != 0) {
        st_munmap(&mm2); st_free(&f1); st_munmap(&mm1); return 0;
    }

    /* Compare */
    if (f1.num_tensors == f2.num_tensors) {
        tw1 = st_get_tensor(&f1, "weight");
        tw2 = st_get_tensor(&f2, "weight");
        if (tw1 && tw2 && tw1->size == tw2->size) {
            d1 = (float *)st_get_tensor_data(&f1, tw1);
            d2 = (float *)st_get_tensor_data(&f2, tw2);
            ok = (memcmp(d1, d2, (size_t)tw1->size) == 0);
        }
    }

    st_free(&f2); st_munmap(&mm2);
    st_free(&f1); st_munmap(&mm1);
    remove(out_path);
    return ok;
}

static int test_parse_too_small(void) {
    /* File smaller than 8 bytes should fail to parse */
    const char *path = "/tmp/tensio_test_tiny.safetensors";
    st_mmap_t mm;
    st_file_t f;
    int ok;
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    fwrite("tiny", 1, 4, fp);
    fclose(fp);

    if (st_mmap(path, &mm) != 0) { remove(path); return 0; }
    ok = (st_parse(&mm, &f) != 0);
    st_munmap(&mm);
    remove(path);
    return ok;
}

static int test_parse_empty_tensors(void) {
    /* File with empty JSON object {} should return 0 tensors */
    const char *path = "/tmp/tensio_test_empty.safetensors";
    st_mmap_t mm;
    st_file_t f;
    int ok;
    FILE *fp;
    const char *json = "{}";
    uint64_t json_len = (uint64_t)strlen(json);

    fp = fopen(path, "wb");
    if (!fp) return 0;
    fwrite(&json_len, 8, 1, fp);
    fwrite(json, 1, (size_t)json_len, fp);
    fclose(fp);

    if (st_mmap(path, &mm) != 0) { remove(path); return 0; }
    if (st_parse(&mm, &f) != 0) { st_munmap(&mm); remove(path); return 0; }
    ok = (f.num_tensors == 0);
    st_free(&f);
    st_munmap(&mm);
    remove(path);
    return ok;
}

/* ----------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------- */

int main(void) {
    printf("=== Safetensors Tests ===\n");

    if (write_fixture() != 0) {
        fprintf(stderr, "Failed to write test fixture\n");
        return 1;
    }

    TEST(mmap_valid_file);
    TEST(mmap_nonexistent);
    TEST(parse_tensor_count);
    TEST(parse_tensor_names);
    TEST(parse_dtype);
    TEST(parse_shape);
    TEST(parse_data_offsets);
    TEST(get_tensor_data);
    TEST(get_tensor_missing);
    TEST(write_roundtrip);
    TEST(parse_too_small);
    TEST(parse_empty_tensors);

    remove(FIXTURE_PATH);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
