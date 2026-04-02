/* test_convert.c — tests for format detection and all conversion paths */

#include "safetensors.h"
#include "gguf.h"
#include "tq.h"
#include "convert.h"
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
 * Fixture helpers
 * ---------------------------------------------------------------- */

static const char *ST_PATH = "/tmp/tensio_conv_test.safetensors";
static const char *GGUF_PATH = "/tmp/tensio_conv_test.gguf";
static const char *TQ_PATH = "/tmp/tensio_conv_test.tq";

static void fwrite_u32(FILE *fp, uint32_t v) { fwrite(&v, 4, 1, fp); }
static void fwrite_u64(FILE *fp, uint64_t v) { fwrite(&v, 8, 1, fp); }

static int create_st_fixture(void) {
    FILE *fp;
    const char *json =
        "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,3],"
        "\"data_offsets\":[0,24]}}";
    uint64_t json_len = (uint64_t)strlen(json);
    float data[6] = {-1.0f, 0.0f, 1.0f, -0.5f, 0.5f, 2.0f};

    fp = fopen(ST_PATH, "wb");
    if (!fp) return -1;
    fwrite(&json_len, 8, 1, fp);
    fwrite(json, 1, (size_t)json_len, fp);
    fwrite(data, sizeof(float), 6, fp);
    fclose(fp);
    return 0;
}

static int create_gguf_fixture(void) {
    FILE *fp;
    float data[6] = {-1.0f, 0.0f, 1.0f, -0.5f, 0.5f, 2.0f};
    long pos, aligned;

    fp = fopen(GGUF_PATH, "wb");
    if (!fp) return -1;

    fwrite_u32(fp, 0x46554747u);
    fwrite_u32(fp, 3);
    fwrite_u64(fp, 1);
    fwrite_u64(fp, 0);

    /* tensor "w": F32 [2,3] */
    fwrite_u64(fp, 1); fwrite("w", 1, 1, fp);
    fwrite_u32(fp, 2);
    fwrite_u64(fp, 2);
    fwrite_u64(fp, 3);
    fwrite_u32(fp, 0); /* F32 */
    fwrite_u64(fp, 0);

    pos = ftell(fp);
    aligned = (pos + 63) & ~63L;
    while (pos < aligned) { uint8_t z = 0; fwrite(&z, 1, 1, fp); pos++; }

    fwrite(data, sizeof(float), 6, fp);
    fclose(fp);
    return 0;
}

static int create_tq_fixture(void) {
    FILE *fp;
    tq_header_t hdr;
    tq_tensor_t desc;
    long pos, aligned;
    /* 2x3=6 values b=2: 6/4 = 2 bytes (round up)
     * vals: [-1, 0, +1, -1, 0, +1]
     * encoded: [0,1,2,0] = 0x24, [1,2,0,0] = 0x09 (with padding) */
    uint8_t packed[2] = {0x24, 0x09};

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = 0x46555154u;
    hdr.version = 2;
    hdr.tensor_count = 1;

    memset(&desc, 0, sizeof(desc));
    strncpy(desc.name, "w", sizeof(desc.name) - 1);
    desc.b = 2;
    desc.rows = 2;
    desc.cols = 3;
    desc.unpacked_size = 2;

    pos = (long)(sizeof(hdr) + sizeof(desc));
    aligned = (pos + 63) & ~63L;
    hdr.data_offset = (uint64_t)aligned;
    hdr.total_data_size = 2;

    fp = fopen(TQ_PATH, "wb");
    if (!fp) return -1;
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fwrite(&desc, sizeof(desc), 1, fp);
    pos = ftell(fp);
    while (pos < aligned) { uint8_t z = 0; fwrite(&z, 1, 1, fp); pos++; }
    fwrite(packed, 1, 2, fp);
    fclose(fp);
    return 0;
}

/* ----------------------------------------------------------------
 * Identity conversion tests
 * ---------------------------------------------------------------- */

static int test_identity_safetensors(void) {
    const char *out = "/tmp/tensio_conv_id.safetensors";
    int ok;
    ok = (convert_any_to_any(ST_PATH, out) == 0);
    if (ok) {
        st_mmap_t mm; st_file_t f;
        if (st_mmap(out, &mm) == 0 && st_parse(&mm, &f) == 0) {
            ok = (f.num_tensors == 1);
            st_free(&f); st_munmap(&mm);
        } else ok = 0;
    }
    remove(out);
    return ok;
}

static int test_identity_gguf(void) {
    const char *out = "/tmp/tensio_conv_id.gguf";
    int ok;
    ok = (convert_any_to_any(GGUF_PATH, out) == 0);
    if (ok) {
        gguf_mmap_t mm; gguf_file_t f;
        if (gguf_mmap(out, &mm) == 0 && gguf_parse(&mm, &f) == 0) {
            ok = (f.tensor_count == 1);
            gguf_free(&f); gguf_munmap(&mm);
        } else ok = 0;
    }
    remove(out);
    return ok;
}

static int test_identity_tq(void) {
    const char *out = "/tmp/tensio_conv_id.tq";
    int ok;
    ok = (convert_any_to_any(TQ_PATH, out) == 0);
    if (ok) {
        tq_file_t f;
        if (tq_mmap(out, &f) == 0) {
            ok = (f.hdr->tensor_count == 1);
            tq_munmap(&f);
        } else ok = 0;
    }
    remove(out);
    return ok;
}

/* ----------------------------------------------------------------
 * Cross-format conversion tests
 * ---------------------------------------------------------------- */

static int test_st_to_gguf(void) {
    const char *out = "/tmp/tensio_conv_st2gguf.gguf";
    int ok = 0;
    if (convert_safetensors_to_gguf(ST_PATH, out) == 0) {
        gguf_mmap_t mm; gguf_file_t f;
        if (gguf_mmap(out, &mm) == 0 && gguf_parse(&mm, &f) == 0) {
            const gguf_tensor_t *t = gguf_get_tensor(&f, "w");
            if (t && t->type == GGUF_TYPE_F32 && t->n_dims == 2 &&
                t->ne[0] == 2 && t->ne[1] == 3) {
                float *d = (float *)gguf_get_tensor_data(&f, t);
                ok = (d[0] == -1.0f && d[2] == 1.0f && d[5] == 2.0f);
            }
            gguf_free(&f); gguf_munmap(&mm);
        }
    }
    remove(out);
    return ok;
}

static int test_gguf_to_st(void) {
    const char *out = "/tmp/tensio_conv_gguf2st.safetensors";
    int ok = 0;
    if (convert_gguf_to_safetensors(GGUF_PATH, out) == 0) {
        st_mmap_t mm; st_file_t f;
        if (st_mmap(out, &mm) == 0 && st_parse(&mm, &f) == 0) {
            const st_tensor_t *t = st_get_tensor(&f, "w");
            if (t && t->dtype == ST_F32 && t->ndim == 2) {
                float *d = (float *)st_get_tensor_data(&f, t);
                ok = (d[0] == -1.0f && d[5] == 2.0f);
            }
            st_free(&f); st_munmap(&mm);
        }
    }
    remove(out);
    return ok;
}

static int test_st_to_tq(void) {
    const char *out = "/tmp/tensio_conv_st2tq.tq";
    int ok = 0;
    if (convert_safetensors_to_tq(ST_PATH, out) == 0) {
        tq_file_t f;
        if (tq_mmap(out, &f) == 0) {
            ok = (f.hdr->tensor_count == 1 &&
                  strcmp(f.tensors[0].name, "w") == 0 &&
                  f.tensors[0].b == 2);
            tq_munmap(&f);
        }
    }
    remove(out);
    return ok;
}

static int test_gguf_to_tq(void) {
    const char *out = "/tmp/tensio_conv_gguf2tq.tq";
    int ok = 0;
    if (convert_gguf_to_tq(GGUF_PATH, out) == 0) {
        tq_file_t f;
        if (tq_mmap(out, &f) == 0) {
            ok = (f.hdr->tensor_count == 1 &&
                  strcmp(f.tensors[0].name, "w") == 0 &&
                  f.tensors[0].b == 2);
            tq_munmap(&f);
        }
    }
    remove(out);
    return ok;
}

static int test_tq_to_st(void) {
    const char *out = "/tmp/tensio_conv_tq2st.safetensors";
    int ok = 0;
    if (convert_tq_to_safetensors(TQ_PATH, out) == 0) {
        st_mmap_t mm; st_file_t f;
        if (st_mmap(out, &mm) == 0 && st_parse(&mm, &f) == 0) {
            const st_tensor_t *t = st_get_tensor(&f, "w");
            if (t && t->dtype == ST_F32 && t->ndim == 2) {
                float *d = (float *)st_get_tensor_data(&f, t);
                /* TQ b=2 dequant: 0→-1, 1→0, 2→+1 */
                ok = (d[0] == -1.0f && d[1] == 0.0f && d[2] == 1.0f);
            }
            st_free(&f); st_munmap(&mm);
        }
    }
    remove(out);
    return ok;
}

static int test_tq_to_gguf(void) {
    const char *out = "/tmp/tensio_conv_tq2gguf.gguf";
    int ok = 0;
    if (convert_tq_to_gguf(TQ_PATH, out) == 0) {
        gguf_mmap_t mm; gguf_file_t f;
        if (gguf_mmap(out, &mm) == 0 && gguf_parse(&mm, &f) == 0) {
            const gguf_tensor_t *t = gguf_get_tensor(&f, "w");
            if (t && t->type == GGUF_TYPE_F32) {
                float *d = (float *)gguf_get_tensor_data(&f, t);
                ok = (d[0] == -1.0f && d[1] == 0.0f && d[2] == 1.0f);
            }
            gguf_free(&f); gguf_munmap(&mm);
        }
    }
    remove(out);
    return ok;
}

/* ----------------------------------------------------------------
 * Round-trip tests: A → B → A, verify data preserved
 * ---------------------------------------------------------------- */

static int test_roundtrip_st_gguf_st(void) {
    const char *mid = "/tmp/tensio_conv_rt_mid.gguf";
    const char *out = "/tmp/tensio_conv_rt_out.safetensors";
    int ok = 0;

    if (convert_safetensors_to_gguf(ST_PATH, mid) == 0 &&
        convert_gguf_to_safetensors(mid, out) == 0) {
        st_mmap_t mm1, mm2; st_file_t f1, f2;
        if (st_mmap(ST_PATH, &mm1) == 0 && st_parse(&mm1, &f1) == 0 &&
            st_mmap(out, &mm2) == 0 && st_parse(&mm2, &f2) == 0) {
            const st_tensor_t *t1 = st_get_tensor(&f1, "w");
            const st_tensor_t *t2 = st_get_tensor(&f2, "w");
            if (t1 && t2 && t1->size == t2->size) {
                float *d1 = (float *)st_get_tensor_data(&f1, t1);
                float *d2 = (float *)st_get_tensor_data(&f2, t2);
                ok = (memcmp(d1, d2, (size_t)t1->size) == 0);
            }
            st_free(&f2); st_munmap(&mm2);
            st_free(&f1); st_munmap(&mm1);
        }
    }
    remove(mid); remove(out);
    return ok;
}

/* ----------------------------------------------------------------
 * Error handling tests
 * ---------------------------------------------------------------- */

static int test_convert_nonexistent(void) {
    return convert_any_to_any("/tmp/no_such_file.safetensors",
                              "/tmp/out.gguf") != 0;
}

static int test_convert_unknown_ext(void) {
    return convert_any_to_any(ST_PATH, "/tmp/out.xyz") != 0;
}

/* ----------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------- */

int main(void) {
    printf("=== Convert Tests ===\n");

    if (create_st_fixture() != 0 ||
        create_gguf_fixture() != 0 ||
        create_tq_fixture() != 0) {
        fprintf(stderr, "Failed to create fixtures\n");
        return 1;
    }

    /* Identity */
    TEST(identity_safetensors);
    TEST(identity_gguf);
    TEST(identity_tq);

    /* Cross-format */
    TEST(st_to_gguf);
    TEST(gguf_to_st);
    TEST(st_to_tq);
    TEST(gguf_to_tq);
    TEST(tq_to_st);
    TEST(tq_to_gguf);

    /* Round-trip */
    TEST(roundtrip_st_gguf_st);

    /* Error handling */
    TEST(convert_nonexistent);
    TEST(convert_unknown_ext);

    remove(ST_PATH);
    remove(GGUF_PATH);
    remove(TQ_PATH);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
