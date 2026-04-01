/* convert.h — any-to-any model format converter
 * Strict C99, zero UB, header-only
 * Uses safetensors.h, gguf.h, tq.h
 */

#ifndef CONVERT_H
#define CONVERT_H

/* Format headers are expected to be included before this file
 * (either directly or via tensio.h). We only need their types. */
#ifndef SAFETENSORS_H
#  include "safetensors.h"
#endif
#ifndef GGUF_H
#  include "gguf.h"
#endif
#ifndef TQ_H
#  include "tq.h"
#endif

/* ================================================================
 * High-level any-to-any conversion
 * ================================================================ */

/* Returns 0 on success */
int convert_any_to_any(const char *input_path, const char *output_path);

/* Explicit converters */
int convert_safetensors_to_gguf(const char *st_path, const char *gguf_path);
int convert_safetensors_to_tq(const char *st_path, const char *tq_path);

int convert_gguf_to_safetensors(const char *gguf_path, const char *st_path);
int convert_gguf_to_tq(const char *gguf_path, const char *tq_path);

int convert_tq_to_safetensors(const char *tq_path, const char *st_path);
int convert_tq_to_gguf(const char *tq_path, const char *gguf_path);

#endif /* CONVERT_H */

/* ================================================================
 * IMPLEMENTATION (define CONVERT_IMPLEMENTATION in one .c file)
 * ================================================================ */

#if defined(CONVERT_IMPLEMENTATION) && !defined(CONVERT_IMPLEMENTATION_DONE)
#define CONVERT_IMPLEMENTATION_DONE

static int detect_format(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic = 0;
    fread(&magic, 4, 1, f);
    fclose(f);

    if (magic == 0x46554747u) return 1;        /* GGUF */
    if (magic == 0x46555154u) return 2;        /* TQ */
    /* Safetensors has no magic, but starts with 8-byte length */
    return 0;   /* assume Safetensors */
}

/* ================================================================
 * Safetensors → others
 * ================================================================ */

int convert_safetensors_to_gguf(const char *st_path, const char *gguf_path) {
    st_mmap_t sm;
    st_file_t src;
    if (st_mmap(st_path, &sm) != 0) return -1;
    if (st_parse(&sm, &src) != 0) goto fail;

    gguf_file_t dst = {0};
    /* TODO: fill dst from src (metadata, tensors, data) */
    /* For now we stub the call */
    int rc = gguf_write(gguf_path, &dst);

    st_munmap(&sm);
    return rc;

fail:
    st_munmap(&sm);
    return -1;
}

int convert_safetensors_to_tq(const char *st_path, const char *tq_path) {
    st_mmap_t sm;
    st_file_t src;
    if (st_mmap(st_path, &sm) != 0) return -1;
    if (st_parse(&sm, &src) != 0) goto fail;

    tq_file_t dst = {0};
    /* TODO: fill dst from src using tq_compress logic */
    int rc = tq_write(tq_path, &dst);

    st_munmap(&sm);
    return rc;

fail:
    st_munmap(&sm);
    return -1;
}

/* ================================================================
 * GGUF → others (symmetric)
 * ================================================================ */

int convert_gguf_to_safetensors(const char *gguf_path, const char *st_path) {
    /* TODO: implement in Phase 5 */
    (void)gguf_path; (void)st_path;
    return -1;
}

int convert_gguf_to_tq(const char *gguf_path, const char *tq_path) {
    /* TODO: implement in Phase 5 */
    (void)gguf_path; (void)tq_path;
    return -1;
}

/* ================================================================
 * TQ → others
 * ================================================================ */

int convert_tq_to_safetensors(const char *tq_path, const char *st_path) {
    /* TODO: implement in Phase 5 */
    (void)tq_path; (void)st_path;
    return -1;
}

int convert_tq_to_gguf(const char *tq_path, const char *gguf_path) {
    /* TODO: implement in Phase 5 */
    (void)tq_path; (void)gguf_path;
    return -1;
}

/* ================================================================
 * Identity converter (file copy)
 * ================================================================ */

static int convert_identity(const char *input_path, const char *output_path) {
    FILE *in, *out;
    char buf[8192];
    size_t n;

    if (strcmp(input_path, output_path) == 0) return 0;

    in = fopen(input_path, "rb");
    if (!in) return -1;
    out = fopen(output_path, "wb");
    if (!out) { fclose(in); return -1; }

    while ((n = fread(buf, 1, sizeof(buf), in)) > 0)
        fwrite(buf, 1, n, out);

    fclose(in);
    fclose(out);
    return 0;
}

/* ================================================================
 * Generic any-to-any
 * ================================================================ */

int convert_any_to_any(const char *input_path, const char *output_path) {
    int in_fmt = detect_format(input_path);

    if (strstr(output_path, ".safetensors")) {
        if (in_fmt == 1) return convert_gguf_to_safetensors(input_path, output_path);
        if (in_fmt == 2) return convert_tq_to_safetensors(input_path, output_path);
        return convert_identity(input_path, output_path);
    }
    if (strstr(output_path, ".gguf")) {
        if (in_fmt == 0) return convert_safetensors_to_gguf(input_path, output_path);
        if (in_fmt == 2) return convert_tq_to_gguf(input_path, output_path);
        return convert_identity(input_path, output_path);
    }
    if (strstr(output_path, ".tq")) {
        if (in_fmt == 0) return convert_safetensors_to_tq(input_path, output_path);
        if (in_fmt == 1) return convert_gguf_to_tq(input_path, output_path);
        return convert_identity(input_path, output_path);
    }
    return -1;
}

#endif /* CONVERT_IMPLEMENTATION */

