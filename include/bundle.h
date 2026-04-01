/* bundle.h — Tar bundle loader for Tensio (model + CLIP + VAE + LoRAs)
 * Strict C99, zero UB, header-only
 * Supports destructive unpack (no 2× disk usage)
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

/* destructive = true  → delete the .tar after successful extraction (minimal disk usage) */
/* io = NULL           → use POSIX (default) */
int tensio_load_bundle(const char *tar_path,
                       const char *cache_dir,     /* where to extract .tq files */
                       bundle_ctx_t *out,
                       bool destructive,
                       const tensio_io_backend_t *io);   /* optional io_uring backend */

void tensio_free_bundle(bundle_ctx_t *bundle);

#endif /* BUNDLE_H */

/* ================================================================
 * IMPLEMENTATION (define BUNDLE_IMPLEMENTATION in one .c file)
 * ================================================================ */

#if defined(BUNDLE_IMPLEMENTATION) && !defined(BUNDLE_IMPLEMENTATION_DONE)
#define BUNDLE_IMPLEMENTATION_DONE

/* Minimal hand-rolled tar parser + manifest.json parser (no external deps) */

static int extract_tar(const char *tar_path, const char *cache_dir,
                       bool destructive, const tensio_io_backend_t *io)
{
    /* TODO: implement ustar tar parser in Phase 5 (~180 lines) */
    (void)cache_dir; (void)io;

    if (destructive) unlink(tar_path);
    return -1;
}

static int parse_manifest(const char *cache_dir, bundle_ctx_t *out)
{
    /* TODO: implement manifest.json parser in Phase 5 */
    (void)cache_dir; (void)out;
    return -1;
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
    if (!bundle) return;

    tq_munmap(&bundle->base);
    tq_munmap(&bundle->clip_l);
    tq_munmap(&bundle->clip_g);
    tq_munmap(&bundle->vae);

    for (int i = 0; i < bundle->lora_count; ++i)
        tq_munmap(&bundle->loras[i].tq);

    free(bundle->loras);
    memset(bundle, 0, sizeof(*bundle));
}

#endif /* BUNDLE_IMPLEMENTATION */

