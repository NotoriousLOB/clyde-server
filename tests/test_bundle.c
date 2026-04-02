/* test_bundle.c — tests for tar bundle extraction and manifest parsing */

#define _POSIX_C_SOURCE 200809L

#include "tq.h"
#include "bundle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-50s ", #name); \
    if (test_##name()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

/* ----------------------------------------------------------------
 * Helper: create a minimal ustar tar containing:
 *   manifest.json
 *   model.tq
 *
 * Tar format: 512-byte header blocks + data blocks (padded to 512)
 * ---------------------------------------------------------------- */

static const char *TAR_PATH = "/tmp/tensio_test_bundle.tar";
static const char *CACHE_DIR = "/tmp/tensio_test_bundle_cache";

static void tar_write_header(FILE *fp, const char *name, uint64_t size) {
    uint8_t hdr[512];
    unsigned int checksum = 0;
    int i;
    char size_oct[12];

    memset(hdr, 0, 512);

    /* name */
    strncpy((char *)hdr, name, 100);

    /* mode */
    memcpy(hdr + 100, "0000644", 7);

    /* uid, gid */
    memcpy(hdr + 108, "0001000", 7);
    memcpy(hdr + 116, "0001000", 7);

    /* size (octal) */
    snprintf(size_oct, sizeof(size_oct), "%011llo", (unsigned long long)size);
    memcpy(hdr + 124, size_oct, 11);

    /* mtime */
    memcpy(hdr + 136, "00000000000", 11);

    /* typeflag = regular file */
    hdr[156] = '0';

    /* ustar magic */
    memcpy(hdr + 257, "ustar", 5);
    memcpy(hdr + 263, "00", 2);

    /* Compute checksum: treat checksum field as spaces */
    memset(hdr + 148, ' ', 8);
    for (i = 0; i < 512; ++i)
        checksum += hdr[i];
    snprintf((char *)hdr + 148, 7, "%06o", checksum);
    hdr[155] = '\0';

    fwrite(hdr, 1, 512, fp);
}

static void tar_write_data(FILE *fp, const void *data, size_t size) {
    size_t padded;
    fwrite(data, 1, size, fp);
    /* Pad to 512-byte boundary */
    padded = ((size + 511) / 512) * 512;
    if (padded > size) {
        uint8_t zeros[512];
        memset(zeros, 0, sizeof(zeros));
        fwrite(zeros, 1, padded - size, fp);
    }
}

static int create_tar_fixture(void) {
    FILE *fp;
    const char *manifest = "{\"base\": \"model.tq\"}";
    size_t manifest_len = strlen(manifest);

    /* Create a minimal TQ file in memory */
    tq_header_t hdr;
    tq_tensor_t desc;
    uint8_t packed[4] = {0x24, 0x24, 0x24, 0x24};
    uint8_t tq_buf[2048];
    size_t tq_size;
    long pos, aligned;

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = TQ_MAGIC;
    hdr.version = TQ_VERSION;
    hdr.tensor_count = 1;
    hdr.model_family_id = TQ_FAMILY_UNKNOWN;

    memset(&desc, 0, sizeof(desc));
    strncpy(desc.name, "weight", sizeof(desc.name) - 1);
    desc.b = 2;
    desc.rows = 4;
    desc.cols = 4;
    desc.unpacked_size = 4;

    pos = (long)(sizeof(hdr) + sizeof(desc));
    aligned = (pos + 63) & ~63L;
    hdr.data_offset = (uint64_t)aligned;
    hdr.total_data_size = 4;

    memset(tq_buf, 0, sizeof(tq_buf));
    memcpy(tq_buf, &hdr, sizeof(hdr));
    memcpy(tq_buf + sizeof(hdr), &desc, sizeof(desc));
    memcpy(tq_buf + aligned, packed, 4);
    tq_size = (size_t)aligned + 4;

    /* Write tar */
    fp = fopen(TAR_PATH, "wb");
    if (!fp) return -1;

    /* Entry 1: manifest.json */
    tar_write_header(fp, "manifest.json", manifest_len);
    tar_write_data(fp, manifest, manifest_len);

    /* Entry 2: model.tq */
    tar_write_header(fp, "model.tq", tq_size);
    tar_write_data(fp, tq_buf, tq_size);

    /* End-of-archive: two zero blocks */
    {
        uint8_t zeros[1024];
        memset(zeros, 0, sizeof(zeros));
        fwrite(zeros, 1, 1024, fp);
    }

    fclose(fp);
    return 0;
}

static void cleanup_cache(void) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/manifest.json", CACHE_DIR);
    remove(path);
    snprintf(path, sizeof(path), "%s/model.tq", CACHE_DIR);
    remove(path);
    rmdir(CACHE_DIR);
}

/* ----------------------------------------------------------------
 * Tests
 * ---------------------------------------------------------------- */

static int test_extract_and_load(void) {
    bundle_ctx_t bundle;
    int ok;

    cleanup_cache();

    ok = (tensio_load_bundle(TAR_PATH, CACHE_DIR, &bundle, false, NULL) == 0);
    if (ok) {
        /* Verify base model was loaded */
        ok = (bundle.base.hdr != NULL &&
              bundle.base.hdr->magic == TQ_MAGIC &&
              bundle.base.hdr->tensor_count == 1);
        tensio_free_bundle(&bundle);
    }
    cleanup_cache();
    return ok;
}

static int test_destructive_unpack(void) {
    bundle_ctx_t bundle;
    struct stat st;
    int ok;

    /* Create a copy of the tar for destructive test */
    const char *tar_copy = "/tmp/tensio_test_bundle_destr.tar";
    {
        FILE *in = fopen(TAR_PATH, "rb");
        FILE *out = fopen(tar_copy, "wb");
        char buf[4096];
        size_t n;
        if (!in || !out) {
            if (in) fclose(in);
            if (out) fclose(out);
            return 0;
        }
        while ((n = fread(buf, 1, sizeof(buf), in)) > 0)
            fwrite(buf, 1, n, out);
        fclose(in);
        fclose(out);
    }

    cleanup_cache();

    ok = (tensio_load_bundle(tar_copy, CACHE_DIR, &bundle, true, NULL) == 0);
    if (ok) {
        /* Tar should have been deleted */
        ok = (stat(tar_copy, &st) != 0);
        tensio_free_bundle(&bundle);
    }

    remove(tar_copy);
    cleanup_cache();
    return ok;
}

static int test_nonexistent_tar(void) {
    bundle_ctx_t bundle;
    return tensio_load_bundle("/tmp/no_such_bundle.tar",
                              CACHE_DIR, &bundle, false, NULL) != 0;
}

static int test_extracted_files_exist(void) {
    bundle_ctx_t bundle;
    struct stat st;
    char path[1024];
    int ok;

    cleanup_cache();

    ok = (tensio_load_bundle(TAR_PATH, CACHE_DIR, &bundle, false, NULL) == 0);
    if (ok) {
        snprintf(path, sizeof(path), "%s/manifest.json", CACHE_DIR);
        ok = (stat(path, &st) == 0 && st.st_size > 0);
        if (ok) {
            snprintf(path, sizeof(path), "%s/model.tq", CACHE_DIR);
            ok = (stat(path, &st) == 0 && st.st_size > 0);
        }
        tensio_free_bundle(&bundle);
    }
    cleanup_cache();
    return ok;
}

static int test_tensor_accessible(void) {
    bundle_ctx_t bundle;
    int ok = 0;

    cleanup_cache();

    if (tensio_load_bundle(TAR_PATH, CACHE_DIR, &bundle, false, NULL) == 0) {
        if (bundle.base.hdr && bundle.base.tensors) {
            float dst[16];
            memset(dst, 0, sizeof(dst));
            tq_dequant(&bundle.base, 0, dst);
            /* byte 0x24 = vals [0,1,2,0] -> [-1, 0, +1, -1] */
            ok = (dst[0] == -1.0f && dst[1] == 0.0f && dst[2] == 1.0f);
        }
        tensio_free_bundle(&bundle);
    }
    cleanup_cache();
    return ok;
}

/* ----------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------- */

int main(void) {
    printf("=== Bundle Tests ===\n");

    if (create_tar_fixture() != 0) {
        fprintf(stderr, "Failed to create tar fixture\n");
        return 1;
    }

    TEST(extract_and_load);
    TEST(destructive_unpack);
    TEST(nonexistent_tar);
    TEST(extracted_files_exist);
    TEST(tensor_accessible);

    remove(TAR_PATH);
    cleanup_cache();

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
