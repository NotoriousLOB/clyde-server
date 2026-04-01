/* vtable_gguf.h
 * Full SQLite3 Virtual Table extension for GGUF files
 * Strict C99, zero UB, header-only
 * Re-uses gguf.h (include before this file)
 */

#ifndef GGUF_VTABLE_H
#define GGUF_VTABLE_H

#include "gguf.h"
#include <sqlite3ext.h>

#ifndef TENSIO_SQLITE_EXT_INIT_DONE
#define TENSIO_SQLITE_EXT_INIT_DONE
SQLITE_EXTENSION_INIT1
#endif

/* ================================================================
 * Virtual table base
 * ================================================================ */

typedef struct {
    sqlite3_vtab base;
    gguf_file_t *model;
    const char  *prefix;
} gguf_vtab_base;

typedef gguf_vtab_base gguf_metadata_vtab;
typedef gguf_vtab_base gguf_tensors_vtab;
typedef gguf_vtab_base gguf_data_vtab;

typedef struct {
    sqlite3_vtab_cursor base;
    gguf_file_t        *model;
    int64_t             rowid;
    int                 eof;
} gguf_vtab_cursor;

int gguf_register_vtables(sqlite3 *db, gguf_file_t *model,
                          const char *prefix);

#endif /* GGUF_VTABLE_H */

/* ================================================================
 * IMPLEMENTATION
 * ================================================================ */

#ifdef GGUF_VTABLE_IMPLEMENTATION

/* ----------------------------------------------------------------
 * Shared helpers
 * ---------------------------------------------------------------- */

static int gguf_vtab_best_index(sqlite3_vtab *pVTab,
                                sqlite3_index_info *pIdxInfo) {
    (void)pVTab; (void)pIdxInfo;
    return SQLITE_OK;
}

static int gguf_vtab_disconnect(sqlite3_vtab *pVTab) {
    gguf_vtab_base *p = (gguf_vtab_base *)pVTab;
    if (p->prefix) sqlite3_free((void *)p->prefix);
    sqlite3_free(p);
    return SQLITE_OK;
}

static int gguf_vtab_readonly(sqlite3_vtab *pVTab, int argc,
                              sqlite3_value **argv,
                              sqlite3_int64 *pRowid) {
    (void)pVTab; (void)argc; (void)argv; (void)pRowid;
    return SQLITE_READONLY;
}

/* ================================================================
 * Metadata VTable (one row per KV pair)
 * ================================================================ */

static int gguf_meta_connect(sqlite3 *db, void *pAux,
                             int argc, const char *const *argv,
                             sqlite3_vtab **ppVTab, char **pzErr) {
    gguf_metadata_vtab *p;
    (void)pzErr;
    p = (gguf_metadata_vtab *)sqlite3_malloc(sizeof(*p));
    if (!p) return SQLITE_NOMEM;
    memset(p, 0, sizeof(*p));
    p->model = (gguf_file_t *)pAux;
    p->prefix = (argc > 3 && argv[3])
        ? sqlite3_mprintf("%s", argv[3]) : NULL;

    *ppVTab = (sqlite3_vtab *)p;
    return sqlite3_declare_vtab(db,
        "CREATE TABLE x(key TEXT PRIMARY KEY, value TEXT, type TEXT)");
}

static int gguf_meta_open(sqlite3_vtab *pVTab,
                          sqlite3_vtab_cursor **ppCursor) {
    gguf_vtab_cursor *c;
    c = (gguf_vtab_cursor *)sqlite3_malloc(sizeof(*c));
    if (!c) return SQLITE_NOMEM;
    memset(c, 0, sizeof(*c));
    c->model = ((gguf_metadata_vtab *)pVTab)->model;
    *ppCursor = (sqlite3_vtab_cursor *)c;
    return SQLITE_OK;
}

static int gguf_meta_close(sqlite3_vtab_cursor *pCursor) {
    sqlite3_free(pCursor);
    return SQLITE_OK;
}

static int gguf_meta_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                            const char *idxStr, int argc,
                            sqlite3_value **argv) {
    gguf_vtab_cursor *c = (gguf_vtab_cursor *)pCursor;
    (void)idxNum; (void)idxStr; (void)argc; (void)argv;
    c->rowid = 0;
    c->eof = (c->model->metadata_count == 0) ? 1 : 0;
    return SQLITE_OK;
}

static int gguf_meta_next(sqlite3_vtab_cursor *pCursor) {
    gguf_vtab_cursor *c = (gguf_vtab_cursor *)pCursor;
    c->rowid++;
    if ((uint64_t)c->rowid >= c->model->metadata_count)
        c->eof = 1;
    return SQLITE_OK;
}

static int gguf_meta_eof(sqlite3_vtab_cursor *pCursor) {
    return ((gguf_vtab_cursor *)pCursor)->eof;
}

static const char *gguf_meta_type_name(gguf_meta_type_t t) {
    switch (t) {
        case GGUF_META_UINT8:   return "uint8";
        case GGUF_META_INT8:    return "int8";
        case GGUF_META_UINT16:  return "uint16";
        case GGUF_META_INT16:   return "int16";
        case GGUF_META_UINT32:  return "uint32";
        case GGUF_META_INT32:   return "int32";
        case GGUF_META_FLOAT32: return "float32";
        case GGUF_META_BOOL:    return "bool";
        case GGUF_META_STRING:  return "string";
        case GGUF_META_ARRAY:   return "array";
        case GGUF_META_UINT64:  return "uint64";
        case GGUF_META_INT64:   return "int64";
        case GGUF_META_FLOAT64: return "float64";
        default: return "unknown";
    }
}

static int gguf_meta_column(sqlite3_vtab_cursor *pCursor,
                            sqlite3_context *pCtx, int col) {
    gguf_vtab_cursor *c = (gguf_vtab_cursor *)pCursor;
    const gguf_kv_t *kv = &c->model->metadata[c->rowid];
    char buf[256];

    switch (col) {
        case 0: /* key */
            sqlite3_result_text(pCtx, kv->key, -1, SQLITE_STATIC);
            break;
        case 1: /* value */
            switch (kv->type) {
                case GGUF_META_STRING:
                    sqlite3_result_text(pCtx, kv->value.str.data,
                                        (int)kv->value.str.len,
                                        SQLITE_STATIC);
                    break;
                case GGUF_META_UINT32:
                    snprintf(buf, sizeof(buf), "%u", kv->value.u32);
                    sqlite3_result_text(pCtx, buf, -1, SQLITE_TRANSIENT);
                    break;
                case GGUF_META_INT32:
                    snprintf(buf, sizeof(buf), "%d", kv->value.i32);
                    sqlite3_result_text(pCtx, buf, -1, SQLITE_TRANSIENT);
                    break;
                case GGUF_META_FLOAT32:
                    snprintf(buf, sizeof(buf), "%g", (double)kv->value.f32);
                    sqlite3_result_text(pCtx, buf, -1, SQLITE_TRANSIENT);
                    break;
                case GGUF_META_UINT64:
                    snprintf(buf, sizeof(buf), "%llu",
                             (unsigned long long)kv->value.u64);
                    sqlite3_result_text(pCtx, buf, -1, SQLITE_TRANSIENT);
                    break;
                case GGUF_META_INT64:
                    snprintf(buf, sizeof(buf), "%lld",
                             (long long)kv->value.i64);
                    sqlite3_result_text(pCtx, buf, -1, SQLITE_TRANSIENT);
                    break;
                case GGUF_META_BOOL:
                    sqlite3_result_text(pCtx,
                        kv->value.b ? "true" : "false", -1, SQLITE_STATIC);
                    break;
                default:
                    sqlite3_result_null(pCtx);
                    break;
            }
            break;
        case 2: /* type */
            sqlite3_result_text(pCtx, gguf_meta_type_name(kv->type),
                                -1, SQLITE_STATIC);
            break;
    }
    return SQLITE_OK;
}

static int gguf_meta_rowid(sqlite3_vtab_cursor *pCursor,
                           sqlite3_int64 *pRowid) {
    *pRowid = ((gguf_vtab_cursor *)pCursor)->rowid;
    return SQLITE_OK;
}

/* ================================================================
 * Tensors VTable (one row per tensor)
 * ================================================================ */

static int gguf_tensors_connect(sqlite3 *db, void *pAux,
                                int argc, const char *const *argv,
                                sqlite3_vtab **ppVTab, char **pzErr) {
    gguf_tensors_vtab *p;
    (void)pzErr;
    p = (gguf_tensors_vtab *)sqlite3_malloc(sizeof(*p));
    if (!p) return SQLITE_NOMEM;
    memset(p, 0, sizeof(*p));
    p->model = (gguf_file_t *)pAux;
    p->prefix = (argc > 3 && argv[3])
        ? sqlite3_mprintf("%s", argv[3]) : NULL;

    *ppVTab = (sqlite3_vtab *)p;
    return sqlite3_declare_vtab(db,
        "CREATE TABLE x("
        "name TEXT PRIMARY KEY, type TEXT, n_dims INTEGER, "
        "shape TEXT, offset INTEGER, size INTEGER)");
}

static int gguf_tensors_open(sqlite3_vtab *pVTab,
                             sqlite3_vtab_cursor **ppCursor) {
    gguf_vtab_cursor *c;
    c = (gguf_vtab_cursor *)sqlite3_malloc(sizeof(*c));
    if (!c) return SQLITE_NOMEM;
    memset(c, 0, sizeof(*c));
    c->model = ((gguf_tensors_vtab *)pVTab)->model;
    *ppCursor = (sqlite3_vtab_cursor *)c;
    return SQLITE_OK;
}

static int gguf_tensors_close(sqlite3_vtab_cursor *pCursor) {
    sqlite3_free(pCursor);
    return SQLITE_OK;
}

static int gguf_tensors_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                               const char *idxStr, int argc,
                               sqlite3_value **argv) {
    gguf_vtab_cursor *c = (gguf_vtab_cursor *)pCursor;
    (void)idxNum; (void)idxStr; (void)argc; (void)argv;
    c->rowid = 0;
    c->eof = (c->model->tensor_count == 0) ? 1 : 0;
    return SQLITE_OK;
}

static int gguf_tensors_next(sqlite3_vtab_cursor *pCursor) {
    gguf_vtab_cursor *c = (gguf_vtab_cursor *)pCursor;
    c->rowid++;
    if ((uint64_t)c->rowid >= c->model->tensor_count)
        c->eof = 1;
    return SQLITE_OK;
}

static int gguf_tensors_eof(sqlite3_vtab_cursor *pCursor) {
    return ((gguf_vtab_cursor *)pCursor)->eof;
}

static const char *gguf_type_name(gguf_type_t t) {
    switch (t) {
        case GGUF_TYPE_F32:  return "F32";
        case GGUF_TYPE_F16:  return "F16";
        case GGUF_TYPE_Q4_0: return "Q4_0";
        case GGUF_TYPE_Q4_1: return "Q4_1";
        case GGUF_TYPE_Q5_0: return "Q5_0";
        case GGUF_TYPE_Q5_1: return "Q5_1";
        case GGUF_TYPE_Q8_0: return "Q8_0";
        case GGUF_TYPE_Q2_K: return "Q2_K";
        case GGUF_TYPE_Q3_K: return "Q3_K";
        case GGUF_TYPE_Q4_K: return "Q4_K";
        case GGUF_TYPE_Q5_K: return "Q5_K";
        case GGUF_TYPE_Q6_K: return "Q6_K";
        case GGUF_TYPE_Q8_K: return "Q8_K";
        case GGUF_TYPE_I8:   return "I8";
        case GGUF_TYPE_I16:  return "I16";
        case GGUF_TYPE_I32:  return "I32";
        case GGUF_TYPE_I64:  return "I64";
        case GGUF_TYPE_F64:  return "F64";
        default: return "UNKNOWN";
    }
}

static int gguf_tensors_column(sqlite3_vtab_cursor *pCursor,
                               sqlite3_context *pCtx, int col) {
    gguf_vtab_cursor *c = (gguf_vtab_cursor *)pCursor;
    const gguf_tensor_t *t = &c->model->tensors[c->rowid];
    char buf[128];

    switch (col) {
        case 0: /* name */
            sqlite3_result_text(pCtx, t->name, -1, SQLITE_STATIC);
            break;
        case 1: /* type */
            sqlite3_result_text(pCtx, gguf_type_name(t->type), -1,
                                SQLITE_STATIC);
            break;
        case 2: /* n_dims */
            sqlite3_result_int(pCtx, (int)t->n_dims);
            break;
        case 3: { /* shape */
            int len = 0;
            uint32_t d;
            len += snprintf(buf + len, sizeof(buf) - (size_t)len, "[");
            for (d = 0; d < t->n_dims; ++d) {
                if (d > 0)
                    len += snprintf(buf + len, sizeof(buf) - (size_t)len,
                                    ",");
                len += snprintf(buf + len, sizeof(buf) - (size_t)len,
                                "%llu", (unsigned long long)t->ne[d]);
            }
            snprintf(buf + len, sizeof(buf) - (size_t)len, "]");
            sqlite3_result_text(pCtx, buf, -1, SQLITE_TRANSIENT);
            break;
        }
        case 4: /* offset */
            sqlite3_result_int64(pCtx, (sqlite3_int64)t->offset);
            break;
        case 5: /* size */
            sqlite3_result_int64(pCtx, (sqlite3_int64)t->size);
            break;
    }
    return SQLITE_OK;
}

static int gguf_tensors_rowid(sqlite3_vtab_cursor *pCursor,
                              sqlite3_int64 *pRowid) {
    *pRowid = ((gguf_vtab_cursor *)pCursor)->rowid;
    return SQLITE_OK;
}

/* ================================================================
 * Tensor Data VTable (zero-copy BLOB)
 * ================================================================ */

static int gguf_data_connect(sqlite3 *db, void *pAux,
                             int argc, const char *const *argv,
                             sqlite3_vtab **ppVTab, char **pzErr) {
    gguf_data_vtab *p;
    (void)pzErr;
    p = (gguf_data_vtab *)sqlite3_malloc(sizeof(*p));
    if (!p) return SQLITE_NOMEM;
    memset(p, 0, sizeof(*p));
    p->model = (gguf_file_t *)pAux;
    p->prefix = (argc > 3 && argv[3])
        ? sqlite3_mprintf("%s", argv[3]) : NULL;

    *ppVTab = (sqlite3_vtab *)p;
    return sqlite3_declare_vtab(db,
        "CREATE TABLE x(tensor_name TEXT, data BLOB)");
}

static int gguf_data_open(sqlite3_vtab *pVTab,
                          sqlite3_vtab_cursor **ppCursor) {
    gguf_vtab_cursor *c;
    c = (gguf_vtab_cursor *)sqlite3_malloc(sizeof(*c));
    if (!c) return SQLITE_NOMEM;
    memset(c, 0, sizeof(*c));
    c->model = ((gguf_data_vtab *)pVTab)->model;
    *ppCursor = (sqlite3_vtab_cursor *)c;
    return SQLITE_OK;
}

static int gguf_data_close(sqlite3_vtab_cursor *pCursor) {
    sqlite3_free(pCursor);
    return SQLITE_OK;
}

static int gguf_data_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                            const char *idxStr, int argc,
                            sqlite3_value **argv) {
    gguf_vtab_cursor *c = (gguf_vtab_cursor *)pCursor;
    (void)idxNum; (void)idxStr; (void)argc; (void)argv;
    c->rowid = 0;
    c->eof = (c->model->tensor_count == 0) ? 1 : 0;
    return SQLITE_OK;
}

static int gguf_data_next(sqlite3_vtab_cursor *pCursor) {
    gguf_vtab_cursor *c = (gguf_vtab_cursor *)pCursor;
    c->rowid++;
    if ((uint64_t)c->rowid >= c->model->tensor_count)
        c->eof = 1;
    return SQLITE_OK;
}

static int gguf_data_eof(sqlite3_vtab_cursor *pCursor) {
    return ((gguf_vtab_cursor *)pCursor)->eof;
}

static int gguf_data_column(sqlite3_vtab_cursor *pCursor,
                            sqlite3_context *pCtx, int col) {
    gguf_vtab_cursor *c = (gguf_vtab_cursor *)pCursor;
    const gguf_tensor_t *t = &c->model->tensors[c->rowid];

    if (col == 0) {
        sqlite3_result_text(pCtx, t->name, -1, SQLITE_STATIC);
    } else if (col == 1) {
        const void *data = gguf_get_tensor_data(c->model, t);
        sqlite3_result_blob(pCtx, data, (int)t->size, SQLITE_STATIC);
    }
    return SQLITE_OK;
}

static int gguf_data_rowid(sqlite3_vtab_cursor *pCursor,
                           sqlite3_int64 *pRowid) {
    *pRowid = ((gguf_vtab_cursor *)pCursor)->rowid;
    return SQLITE_OK;
}

/* ================================================================
 * Module definitions
 * ================================================================ */

static sqlite3_module gguf_metadata_module = {
    .iVersion    = 0,
    .xConnect    = gguf_meta_connect,
    .xBestIndex  = gguf_vtab_best_index,
    .xDisconnect = gguf_vtab_disconnect,
    .xOpen       = gguf_meta_open,
    .xClose      = gguf_meta_close,
    .xFilter     = gguf_meta_filter,
    .xNext       = gguf_meta_next,
    .xEof        = gguf_meta_eof,
    .xColumn     = gguf_meta_column,
    .xRowid      = gguf_meta_rowid,
    .xUpdate     = gguf_vtab_readonly,
};

static sqlite3_module gguf_tensors_module = {
    .iVersion    = 0,
    .xConnect    = gguf_tensors_connect,
    .xBestIndex  = gguf_vtab_best_index,
    .xDisconnect = gguf_vtab_disconnect,
    .xOpen       = gguf_tensors_open,
    .xClose      = gguf_tensors_close,
    .xFilter     = gguf_tensors_filter,
    .xNext       = gguf_tensors_next,
    .xEof        = gguf_tensors_eof,
    .xColumn     = gguf_tensors_column,
    .xRowid      = gguf_tensors_rowid,
    .xUpdate     = gguf_vtab_readonly,
};

static sqlite3_module gguf_data_module = {
    .iVersion    = 0,
    .xConnect    = gguf_data_connect,
    .xBestIndex  = gguf_vtab_best_index,
    .xDisconnect = gguf_vtab_disconnect,
    .xOpen       = gguf_data_open,
    .xClose      = gguf_data_close,
    .xFilter     = gguf_data_filter,
    .xNext       = gguf_data_next,
    .xEof        = gguf_data_eof,
    .xColumn     = gguf_data_column,
    .xRowid      = gguf_data_rowid,
    .xUpdate     = gguf_vtab_readonly,
};

/* ================================================================
 * Registration
 * ================================================================ */

int gguf_register_vtables(sqlite3 *db, gguf_file_t *model,
                          const char *prefix) {
    int rc;
    char *meta_name, *tens_name, *data_name;

    if (prefix) {
        meta_name = sqlite3_mprintf("%s_metadata", prefix);
        tens_name = sqlite3_mprintf("%s_tensors", prefix);
        data_name = sqlite3_mprintf("%s_data", prefix);
    } else {
        meta_name = sqlite3_mprintf("gguf_metadata");
        tens_name = sqlite3_mprintf("gguf_tensors");
        data_name = sqlite3_mprintf("gguf_data");
    }

    rc = sqlite3_create_module_v2(db, meta_name, &gguf_metadata_module,
                                  model, NULL);
    sqlite3_free(meta_name);
    if (rc != SQLITE_OK) { sqlite3_free(tens_name); sqlite3_free(data_name); return rc; }

    rc = sqlite3_create_module_v2(db, tens_name, &gguf_tensors_module,
                                  model, NULL);
    sqlite3_free(tens_name);
    if (rc != SQLITE_OK) { sqlite3_free(data_name); return rc; }

    rc = sqlite3_create_module_v2(db, data_name, &gguf_data_module,
                                  model, NULL);
    sqlite3_free(data_name);
    return rc;
}

#endif /* GGUF_VTABLE_IMPLEMENTATION */
