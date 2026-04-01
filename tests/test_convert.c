/* test_convert.c — tests for format detection and conversion stubs */

#include "safetensors.h"
#include "gguf.h"
#include "tq.h"
#include "convert.h"
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
 * Fixture helpers
 * ---------------------------------------------------------------- */

static const char *ST_PATH = "/tmp/tensio_conv_test.safetensors";
static const char *GGUF_PATH = "/tmp/tensio_conv_test.gguf";
static const char *TQ_PATH = "/tmp/tensio_conv_test.tq";

static void write_u32(FILE *fp, uint32_t v) { fwrite(&v, 4, 1, fp); }
static void write_u64(FILE *fp, uint64_t v) { fwrite(&v, 8, 1, fp); }

static int create_st_fixture(void) {
    FILE *fp;
    const char *json =
        "{\"w\":{\"dtype\":\"F32\",\"shape\":[2],"
        "\"data_offsets\":[0,8]}}";
    uint64_t json_len = (uint64_t)strlen(json);
    float data[2] = {1.0f, 2.0f};

    fp = fopen(ST_PATH, "wb");
    if (!fp) return -1;
    fwrite(&json_len, 8, 1, fp);
    fwrite(json, 1, (size_t)json_len, fp);
    fwrite(data, sizeof(float), 2, fp);
    fclose(fp);
    return 0;
}

static int create_gguf_fixture(void) {
    FILE *fp;
    float data[2] = {3.0f, 4.0f};
    long pos, aligned;

    fp = fopen(GGUF_PATH, "wb");
    if (!fp) return -1;

    write_u32(fp, 0x46554747u);
    write_u32(fp, 3);
    write_u64(fp, 1);
    write_u64(fp, 0);

    /* tensor "w": F32 [2] */
    write_u64(fp, 1); fwrite("w", 1, 1, fp); /* name */
    write_u32(fp, 1); /* n_dims */
    write_u64(fp, 2); /* ne[0] */
    write_u32(fp, 0); /* F32 */
    write_u64(fp, 0); /* offset */

    pos = ftell(fp);
    aligned = (pos + 63) & ~63L;
    while (pos < aligned) { uint8_t z = 0; fwrite(&z, 1, 1, fp); pos++; }

    fwrite(data, sizeof(float), 2, fp);
    fclose(fp);
    return 0;
}

static int create_tq_fixture(void) {
    FILE *fp;
    tq_header_t hdr;
    tq_tensor_t desc;
    long pos, aligned;
    uint8_t packed[1] = {0x24}; /* 4 values b=2 */

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = 0x46555154u;
    hdr.version = 2;
    hdr.tensor_count = 1;

    memset(&desc, 0, sizeof(desc));
    strncpy(desc.name, "w", sizeof(desc.name) - 1);
    desc.b = 2;
    desc.rows = 2;
    desc.cols = 2;
    desc.unpacked_size = 1;

    pos = (long)(sizeof(hdr) + sizeof(desc));
    aligned = (pos + 63) & ~63L;
    hdr.data_offset = (uint64_t)aligned;
    hdr.total_data_size = 1;

    fp = fopen(TQ_PATH, "wb");
    if (!fp) return -1;
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fwrite(&desc, sizeof(desc), 1, fp);
    pos = ftell(fp);
    while (pos < aligned) { uint8_t z = 0; fwrite(&z, 1, 1, fp); pos++; }
    fwrite(packed, 1, 1, fp);
    fclose(fp);
    return 0;
}

/* ----------------------------------------------------------------
 * Tests
 * ---------------------------------------------------------------- */

static int test_identity_safetensors(void) {
    const char *out = "/tmp/tensio_conv_id.safetensors";
    int ok;
    ok = (convert_any_to_any(ST_PATH, out) == 0);
    if (ok) {
        /* Verify the copy is valid */
        st_mmap_t mm;
        st_file_t f;
        if (st_mmap(out, &mm) == 0 && st_parse(&mm, &f) == 0) {
            ok = (f.num_tensors == 1);
            st_free(&f);
            st_munmap(&mm);
        } else {
            ok = 0;
        }
    }
    remove(out);
    return ok;
}

static int test_identity_gguf(void) {
    const char *out = "/tmp/tensio_conv_id.gguf";
    int ok;
    ok = (convert_any_to_any(GGUF_PATH, out) == 0);
    if (ok) {
        gguf_mmap_t mm;
        gguf_file_t f;
        if (gguf_mmap(out, &mm) == 0 && gguf_parse(&mm, &f) == 0) {
            ok = (f.tensor_count == 1);
            gguf_free(&f);
            gguf_munmap(&mm);
        } else {
            ok = 0;
        }
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
        } else {
            ok = 0;
        }
    }
    remove(out);
    return ok;
}

static int test_convert_nonexistent(void) {
    return convert_any_to_any("/tmp/no_such_file.safetensors",
                              "/tmp/out.gguf") != 0;
}

static int test_convert_unknown_ext(void) {
    return convert_any_to_any(ST_PATH, "/tmp/out.xyz") != 0;
}

static int test_st_to_gguf_stub(void) {
    /* Currently st→gguf writes an empty GGUF (stub) — just verify it doesn't crash */
    const char *out = "/tmp/tensio_conv_st2gguf.gguf";
    int rc = convert_safetensors_to_gguf(ST_PATH, out);
    /* rc == 0 means the stub ran without crash */
    remove(out);
    return (rc == 0);
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

    TEST(identity_safetensors);
    TEST(identity_gguf);
    TEST(identity_tq);
    TEST(convert_nonexistent);
    TEST(convert_unknown_ext);
    TEST(st_to_gguf_stub);

    remove(ST_PATH);
    remove(GGUF_PATH);
    remove(TQ_PATH);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
