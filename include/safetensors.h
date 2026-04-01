/* safetensors.h — mmap parser + writer for Safetensors format
 * Strict C99, zero UB, header-only
 * Requires yyjson for JSON parsing
 * -std=c99 -pedantic -Wall -Wextra -Werror -march=native
 */

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/stat.h>
#endif

#include <yyjson.h>

/* ================================================================
 * Public Types
 * ================================================================ */

typedef enum {
    ST_F32, ST_F16, ST_BF16, ST_I8, ST_I16, ST_I32, ST_I64,
    ST_U8, ST_U16, ST_U32, ST_U64, ST_F64
} st_dtype_t;

typedef struct {
    char       *name;
    st_dtype_t  dtype;
    uint32_t    ndim;
    uint64_t    shape[8];
    uint64_t    offset;   /* byte offset into data section */
    uint64_t    size;     /* byte size */
} st_tensor_t;

typedef struct {
    uint8_t    *base;     /* mmap base */
    size_t      size;
    void       *mmap_handle;
} st_mmap_t;

typedef struct {
    uint64_t      header_len;
    char         *header_json;   /* raw JSON (points into mmap, not owned) */
    st_tensor_t  *tensors;       /* owned, allocated by st_parse */
    uint32_t      num_tensors;
    uint8_t      *data;          /* start of raw tensor data (in mmap) */
    uint64_t      data_size;     /* byte size of the data section */
} st_file_t;

/* ================================================================
 * Public API
 * ================================================================ */

int  st_mmap(const char *path, st_mmap_t *out);
void st_munmap(st_mmap_t *m);

int  st_parse(const st_mmap_t *mmap, st_file_t *out);
void st_free(st_file_t *file);

const st_tensor_t *st_get_tensor(const st_file_t *f, const char *name);
void *st_get_tensor_data(const st_file_t *f, const st_tensor_t *t);

int st_write(const char *path, const st_file_t *f);

#endif /* SAFETENSORS_H */

/* ================================================================
 * IMPLEMENTATION (define SAFETENSORS_IMPLEMENTATION in one .c)
 * ================================================================ */

#if defined(SAFETENSORS_IMPLEMENTATION) && !defined(SAFETENSORS_IMPLEMENTATION_DONE)
#define SAFETENSORS_IMPLEMENTATION_DONE

/* ================================================================
 * mmap / munmap
 * ================================================================ */

int st_mmap(const char *path, st_mmap_t *out) {
#ifdef _WIN32
    /* TODO: Windows mmap via CreateFileMapping */
    (void)path; (void)out;
    return -1;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    struct stat sb;
    if (fstat(fd, &sb) != 0) { close(fd); return -1; }

    out->base = (uint8_t *)mmap(NULL, (size_t)sb.st_size, PROT_READ,
                                MAP_PRIVATE, fd, 0);
    close(fd);
    if (out->base == MAP_FAILED) { out->base = NULL; return -1; }

    out->size = (size_t)sb.st_size;
    out->mmap_handle = NULL;
    return 0;
#endif
}

void st_munmap(st_mmap_t *m) {
    if (m->base) munmap(m->base, m->size);
    memset(m, 0, sizeof(*m));
}

/* ================================================================
 * Dtype helpers
 * ================================================================ */

static st_dtype_t st_dtype_from_str(const char *s, size_t len) {
    if (len == 3 && memcmp(s, "F32", 3) == 0) return ST_F32;
    if (len == 3 && memcmp(s, "F16", 3) == 0) return ST_F16;
    if (len == 4 && memcmp(s, "BF16", 4) == 0) return ST_BF16;
    if (len == 2 && memcmp(s, "I8", 2) == 0) return ST_I8;
    if (len == 3 && memcmp(s, "I16", 3) == 0) return ST_I16;
    if (len == 3 && memcmp(s, "I32", 3) == 0) return ST_I32;
    if (len == 3 && memcmp(s, "I64", 3) == 0) return ST_I64;
    if (len == 2 && memcmp(s, "U8", 2) == 0) return ST_U8;
    if (len == 3 && memcmp(s, "U16", 3) == 0) return ST_U16;
    if (len == 3 && memcmp(s, "U32", 3) == 0) return ST_U32;
    if (len == 3 && memcmp(s, "U64", 3) == 0) return ST_U64;
    if (len == 3 && memcmp(s, "F64", 3) == 0) return ST_F64;
    return ST_F32; /* fallback */
}

static const char *st_dtype_to_string(st_dtype_t dt) {
    switch (dt) {
        case ST_F32:  return "F32";
        case ST_F16:  return "F16";
        case ST_BF16: return "BF16";
        case ST_I8:   return "I8";
        case ST_I16:  return "I16";
        case ST_I32:  return "I32";
        case ST_I64:  return "I64";
        case ST_U8:   return "U8";
        case ST_U16:  return "U16";
        case ST_U32:  return "U32";
        case ST_U64:  return "U64";
        case ST_F64:  return "F64";
        default:      return "F32";
    }
}

/* Used by converters and consumers; may appear unused in some TUs */
static size_t st_dtype_size(st_dtype_t dt) {
    switch (dt) {
        case ST_F32: case ST_I32: case ST_U32: return 4;
        case ST_F16: case ST_BF16: case ST_I16: case ST_U16: return 2;
        case ST_I8: case ST_U8: return 1;
        case ST_I64: case ST_U64: case ST_F64: return 8;
        default: return 4;
    }
}

/* ================================================================
 * Parser — uses yyjson
 * ================================================================ */

int st_parse(const st_mmap_t *mmap, st_file_t *out) {
    yyjson_doc *doc;
    yyjson_val *root, *key, *val;
    yyjson_obj_iter iter;
    uint32_t count, idx;

    memset(out, 0, sizeof(*out));

    if (mmap->size < 8) return -1;

    memcpy(&out->header_len, mmap->base, 8);
    if (mmap->size < 8 + out->header_len) return -1;

    out->header_json = (char *)(mmap->base + 8);
    out->data = mmap->base + 8 + out->header_len;
    out->data_size = mmap->size - 8 - out->header_len;

    /* Parse JSON header */
    doc = yyjson_read(out->header_json, (size_t)out->header_len, 0);
    if (!doc) return -1;

    root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) { yyjson_doc_free(doc); return -1; }

    /* First pass: count tensors (skip __metadata__) */
    count = 0;
    yyjson_obj_iter_init(root, &iter);
    while ((key = yyjson_obj_iter_next(&iter)) != NULL) {
        const char *k = yyjson_get_str(key);
        if (k && strcmp(k, "__metadata__") != 0)
            count++;
    }

    if (count == 0) {
        out->num_tensors = 0;
        out->tensors = NULL;
        yyjson_doc_free(doc);
        return 0;
    }

    out->tensors = (st_tensor_t *)calloc(count, sizeof(st_tensor_t));
    if (!out->tensors) { yyjson_doc_free(doc); return -1; }

    /* Second pass: populate tensor array */
    idx = 0;
    yyjson_obj_iter_init(root, &iter);
    while ((key = yyjson_obj_iter_next(&iter)) != NULL) {
        const char *k = yyjson_get_str(key);
        size_t klen;
        yyjson_val *dtype_val, *shape_val, *offsets_val;

        if (!k || strcmp(k, "__metadata__") == 0) continue;

        val = yyjson_obj_iter_get_val(key);
        if (!yyjson_is_obj(val)) continue;

        /* Name */
        klen = yyjson_get_len(key);
        out->tensors[idx].name = (char *)malloc(klen + 1);
        if (!out->tensors[idx].name) continue;
        memcpy(out->tensors[idx].name, k, klen);
        out->tensors[idx].name[klen] = '\0';

        /* dtype */
        dtype_val = yyjson_obj_get(val, "dtype");
        if (dtype_val && yyjson_is_str(dtype_val)) {
            const char *ds = yyjson_get_str(dtype_val);
            size_t dlen = yyjson_get_len(dtype_val);
            out->tensors[idx].dtype = st_dtype_from_str(ds, dlen);
        }

        /* shape */
        shape_val = yyjson_obj_get(val, "shape");
        if (shape_val && yyjson_is_arr(shape_val)) {
            size_t si, sn = yyjson_arr_size(shape_val);
            if (sn > 8) sn = 8;
            out->tensors[idx].ndim = (uint32_t)sn;
            for (si = 0; si < sn; ++si) {
                yyjson_val *dim = yyjson_arr_get(shape_val, si);
                out->tensors[idx].shape[si] =
                    dim ? (uint64_t)yyjson_get_sint(dim) : 0;
            }
        }

        /* data_offsets: [begin, end] */
        offsets_val = yyjson_obj_get(val, "data_offsets");
        if (offsets_val && yyjson_is_arr(offsets_val)) {
            yyjson_val *begin = yyjson_arr_get(offsets_val, 0);
            yyjson_val *end   = yyjson_arr_get(offsets_val, 1);
            uint64_t b = begin ? (uint64_t)yyjson_get_sint(begin) : 0;
            uint64_t e = end   ? (uint64_t)yyjson_get_sint(end)   : 0;
            out->tensors[idx].offset = b;
            out->tensors[idx].size = e - b;
        }

        idx++;
    }

    out->num_tensors = idx;
    yyjson_doc_free(doc);
    return 0;
}

void st_free(st_file_t *file) {
    if (file->tensors) {
        uint32_t i;
        for (i = 0; i < file->num_tensors; ++i)
            free(file->tensors[i].name);
        free(file->tensors);
    }
    memset(file, 0, sizeof(*file));
}

const st_tensor_t *st_get_tensor(const st_file_t *f, const char *name) {
    uint32_t i;
    for (i = 0; i < f->num_tensors; ++i) {
        if (strcmp(f->tensors[i].name, name) == 0)
            return &f->tensors[i];
    }
    return NULL;
}

void *st_get_tensor_data(const st_file_t *f, const st_tensor_t *t) {
    return f->data + t->offset;
}

/* ================================================================
 * Writer — generates JSON header + writes data
 * ================================================================ */

int st_write(const char *path, const st_file_t *f) {
    yyjson_mut_doc *doc;
    yyjson_mut_val *root;
    FILE *fp;
    char *json;
    size_t json_len;
    uint64_t header_len;
    uint32_t i;

    /* Build JSON header */
    doc = yyjson_mut_doc_new(NULL);
    if (!doc) return -1;

    root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    for (i = 0; i < f->num_tensors; ++i) {
        const st_tensor_t *t = &f->tensors[i];
        yyjson_mut_val *tobj = yyjson_mut_obj(doc);
        yyjson_mut_val *shape_arr = yyjson_mut_arr(doc);
        yyjson_mut_val *offsets_arr = yyjson_mut_arr(doc);
        uint32_t d;

        /* dtype */
        yyjson_mut_obj_add_str(doc, tobj, "dtype",
                               st_dtype_to_string(t->dtype));

        /* shape */
        for (d = 0; d < t->ndim; ++d)
            yyjson_mut_arr_add_sint(doc, shape_arr, (int64_t)t->shape[d]);
        yyjson_mut_obj_add_val(doc, tobj, "shape", shape_arr);

        /* data_offsets */
        yyjson_mut_arr_add_sint(doc, offsets_arr, (int64_t)t->offset);
        yyjson_mut_arr_add_sint(doc, offsets_arr,
                                (int64_t)(t->offset + t->size));
        yyjson_mut_obj_add_val(doc, tobj, "data_offsets", offsets_arr);

        yyjson_mut_obj_add_val(doc, root, t->name, tobj);
    }

    json = yyjson_mut_write(doc, 0, &json_len);
    yyjson_mut_doc_free(doc);
    if (!json) return -1;

    /* Write file */
    fp = fopen(path, "wb");
    if (!fp) { free(json); return -1; }

    header_len = (uint64_t)json_len;
    fwrite(&header_len, 8, 1, fp);
    fwrite(json, 1, json_len, fp);
    free(json);

    if (f->data && f->data_size > 0)
        fwrite(f->data, 1, (size_t)f->data_size, fp);

    fclose(fp);
    return 0;
}

#endif /* SAFETENSORS_IMPLEMENTATION */
