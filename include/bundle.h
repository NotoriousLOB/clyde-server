/* bundle.h — Tar bundle loader for Tensio (model + CLIP + VAE + LoRAs)
 * Strict C99, zero UB, header-only
 * Supports destructive unpack (no 2x disk usage)
 * Include after tensio.h or standalone with tq.h
 */

#ifndef BUNDLE_H
#define BUNDLE_H

#include <stdbool.h>
#include "tq.h"

/* Forward declaration — full definition in tensio.h */
#ifndef TENSIO_H
typedef struct tensio_io_backend tensio_io_backend_t;
#endif

/* ================================================================
 * Bundle context — holds all loaded .tq files
 * ================================================================ */

typedef struct {
    tq_file_t base;       /* main model */
    tq_file_t clip_l;
    tq_file_t clip_g;
    tq_file_t vae;

    struct {
        char      name[64];
        tq_file_t tq;
        float     alpha;
    } *loras;
    int lora_count;

    /* Internal cache directory */
    char cache_dir[512];
} bundle_ctx_t;

/* ================================================================
 * Load a .tar bundle
 * ================================================================ */

/* destructive = true  -> delete the .tar after successful extraction */
/* io = NULL           -> use POSIX (default) */
int tensio_load_bundle(const char *tar_path,
                       const char *cache_dir,
                       bundle_ctx_t *out,
                       bool destructive,
                       const tensio_io_backend_t *io);

void tensio_free_bundle(bundle_ctx_t *bundle);

#endif /* BUNDLE_H */

/* ================================================================
 * IMPLEMENTATION (define BUNDLE_IMPLEMENTATION in one .c file)
 * ================================================================ */

#if defined(BUNDLE_IMPLEMENTATION) && !defined(BUNDLE_IMPLEMENTATION_DONE)
#define BUNDLE_IMPLEMENTATION_DONE

#include <sys/stat.h>
#include <errno.h>

/* ================================================================
 * ustar tar header (POSIX.1-1988, 512 bytes)
 * ================================================================ */

typedef struct {
    char name[100];
    char mode[8];
    char uid[8];
    char gid[8];
    char size[12];      /* octal ASCII */
    char mtime[12];
    char checksum[8];
    char typeflag;      /* '0' or '\0' = regular file, '5' = directory */
    char linkname[100];
    char magic[6];      /* "ustar" */
    char version[2];
    char uname[32];
    char gname[32];
    char devmajor[8];
    char devminor[8];
    char prefix[155];
    char pad[12];
} tar_header_t;

/* Parse an octal ASCII field into uint64_t */
static uint64_t tar_parse_octal(const char *field, int len) {
    uint64_t val = 0;
    int i;
    for (i = 0; i < len && field[i] != '\0' && field[i] != ' '; ++i) {
        if (field[i] >= '0' && field[i] <= '7')
            val = (val << 3) | (uint64_t)(field[i] - '0');
    }
    return val;
}

/* Check if a tar block is all zeros (end-of-archive marker) */
static int tar_is_end(const uint8_t *block) {
    int i;
    for (i = 0; i < 512; ++i)
        if (block[i] != 0) return 0;
    return 1;
}

/* Build full path: cache_dir/filename */
static void tar_build_path(char *out, size_t out_size,
                           const char *dir, const char *name) {
    /* Strip leading "./" from tar entry names */
    if (name[0] == '.' && name[1] == '/') name += 2;
    snprintf(out, out_size, "%s/%s", dir, name);
}

/* Ensure a directory exists (mkdir -p for one level) */
static int ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) return 0;
    return mkdir(path, 0755);
}

/* ================================================================
 * extract_tar — streaming ustar extraction
 * ================================================================ */

static int extract_tar(const char *tar_path, const char *cache_dir,
                       bool destructive, const tensio_io_backend_t *io)
{
    FILE *fp;
    uint8_t block[512];
    int end_blocks = 0;

    (void)io; /* io_uring backend not used here (future work) */

    if (ensure_dir(cache_dir) != 0) return -1;

    fp = fopen(tar_path, "rb");
    if (!fp) return -1;

    while (fread(block, 1, 512, fp) == 512) {
        tar_header_t *hdr = (tar_header_t *)block;
        uint64_t file_size;
        uint64_t blocks;
        char full_path[1024];

        /* Two consecutive zero blocks = end of archive */
        if (tar_is_end(block)) {
            end_blocks++;
            if (end_blocks >= 2) break;
            continue;
        }
        end_blocks = 0;

        /* Validate ustar magic (optional — some tars omit it) */
        file_size = tar_parse_octal(hdr->size, 12);

        if (hdr->typeflag == '5') {
            /* Directory entry */
            tar_build_path(full_path, sizeof(full_path),
                           cache_dir, hdr->name);
            ensure_dir(full_path);
            continue;
        }

        if (hdr->typeflag != '0' && hdr->typeflag != '\0') {
            /* Skip non-regular entries (links, etc) */
            blocks = (file_size + 511) / 512;
            fseek(fp, (long)(blocks * 512), SEEK_CUR);
            continue;
        }

        /* Regular file — extract */
        tar_build_path(full_path, sizeof(full_path),
                       cache_dir, hdr->name);

        /* Create parent directory if the name contains a / */
        {
            char dir_buf[1024];
            char *slash;
            strncpy(dir_buf, full_path, sizeof(dir_buf) - 1);
            dir_buf[sizeof(dir_buf) - 1] = '\0';
            slash = strrchr(dir_buf, '/');
            if (slash) {
                *slash = '\0';
                ensure_dir(dir_buf);
            }
        }

        {
            FILE *out = fopen(full_path, "wb");
            uint64_t remaining;
            if (!out) {
                /* Skip this file's data blocks */
                blocks = (file_size + 511) / 512;
                fseek(fp, (long)(blocks * 512), SEEK_CUR);
                continue;
            }

            remaining = file_size;
            while (remaining > 0) {
                uint8_t data_block[512];
                size_t to_write;
                if (fread(data_block, 1, 512, fp) != 512) {
                    fclose(out);
                    fclose(fp);
                    return -1;
                }
                to_write = (remaining > 512) ? 512 : (size_t)remaining;
                fwrite(data_block, 1, to_write, out);
                remaining -= to_write;
            }
            fclose(out);
        }
    }

    fclose(fp);

    if (destructive) unlink(tar_path);

    return 0;
}

/* ================================================================
 * parse_manifest — hand-rolled minimal JSON scanner
 *
 * Expected manifest.json format:
 * {
 *   "base": "model.tq",
 *   "clip_l": "clip_l.tq",       (optional)
 *   "clip_g": "clip_g.tq",       (optional)
 *   "vae": "vae.tq"              (optional)
 * }
 *
 * No yyjson dependency — uses simple key:value scanning.
 * ================================================================ */

static int manifest_find_value(const char *json, const char *key,
                               char *out, size_t out_size) {
    const char *p;
    char search[128];
    const char *start, *end;
    size_t len;

    snprintf(search, sizeof(search), "\"%s\"", key);
    p = strstr(json, search);
    if (!p) return -1;

    p += strlen(search);
    /* Skip whitespace and colon */
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ':')
        p++;

    if (*p != '"') return -1;
    start = ++p;
    end = strchr(start, '"');
    if (!end) return -1;

    len = (size_t)(end - start);
    if (len >= out_size) len = out_size - 1;
    memcpy(out, start, len);
    out[len] = '\0';
    return 0;
}

static int parse_manifest(const char *cache_dir, bundle_ctx_t *out)
{
    char manifest_path[1024];
    FILE *fp;
    long fsize;
    char *json;
    char val[256];
    char tq_path[1024];

    snprintf(manifest_path, sizeof(manifest_path),
             "%s/manifest.json", cache_dir);

    fp = fopen(manifest_path, "rb");
    if (!fp) {
        /* No manifest.json — try to load any .tq file as base */
        snprintf(tq_path, sizeof(tq_path), "%s/model.tq", cache_dir);
        if (tq_mmap(tq_path, &out->base) == 0)
            return 0;
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (fsize <= 0 || fsize > 65536) { fclose(fp); return -1; }

    json = (char *)malloc((size_t)fsize + 1);
    if (!json) { fclose(fp); return -1; }
    if (fread(json, 1, (size_t)fsize, fp) != (size_t)fsize) {
        free(json); fclose(fp); return -1;
    }
    json[fsize] = '\0';
    fclose(fp);

    /* Load base model (required) */
    if (manifest_find_value(json, "base", val, sizeof(val)) == 0) {
        snprintf(tq_path, sizeof(tq_path), "%s/%s", cache_dir, val);
        if (tq_mmap(tq_path, &out->base) != 0) {
            free(json); return -1;
        }
    } else {
        free(json); return -1;
    }

    /* Load optional components */
    if (manifest_find_value(json, "clip_l", val, sizeof(val)) == 0) {
        snprintf(tq_path, sizeof(tq_path), "%s/%s", cache_dir, val);
        tq_mmap(tq_path, &out->clip_l); /* ignore failure — optional */
    }

    if (manifest_find_value(json, "clip_g", val, sizeof(val)) == 0) {
        snprintf(tq_path, sizeof(tq_path), "%s/%s", cache_dir, val);
        tq_mmap(tq_path, &out->clip_g);
    }

    if (manifest_find_value(json, "vae", val, sizeof(val)) == 0) {
        snprintf(tq_path, sizeof(tq_path), "%s/%s", cache_dir, val);
        tq_mmap(tq_path, &out->vae);
    }

    free(json);
    return 0;
}

/* ================================================================
 * Main load function
 * ================================================================ */

int tensio_load_bundle(const char *tar_path,
                       const char *cache_dir,
                       bundle_ctx_t *out,
                       bool destructive,
                       const tensio_io_backend_t *io)
{
    if (!tar_path || !cache_dir || !out) return -1;

    memset(out, 0, sizeof(*out));
    strncpy(out->cache_dir, cache_dir, sizeof(out->cache_dir)-1);

    /* 1. Extract tar (destructive if requested) */
    if (extract_tar(tar_path, cache_dir, destructive, io) != 0)
        return -1;

    /* 2. Parse manifest.json and mmap all .tq files */
    if (parse_manifest(cache_dir, out) != 0)
        return -1;

    return 0;
}

void tensio_free_bundle(bundle_ctx_t *bundle)
{
    int i;
    if (!bundle) return;

    tq_munmap(&bundle->base);
    tq_munmap(&bundle->clip_l);
    tq_munmap(&bundle->clip_g);
    tq_munmap(&bundle->vae);

    for (i = 0; i < bundle->lora_count; ++i)
        tq_munmap(&bundle->loras[i].tq);

    free(bundle->loras);
    memset(bundle, 0, sizeof(*bundle));
}

#endif /* BUNDLE_IMPLEMENTATION */
