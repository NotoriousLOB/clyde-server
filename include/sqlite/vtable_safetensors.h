/* vtable_safetensors.h
 * Full SQLite3 Virtual Table extension for Safetensors files
 * Strict C99, zero UB, header-only
 * Requires yyjson (-lyyjson) and safetensors.h
 */

#ifndef SAFETENSORS_VTABLE_H
#define SAFETENSORS_VTABLE_H

#include "safetensors.h"
#include <sqlite3ext.h>
#include <yyjson.h>

#ifndef TENSIO_SQLITE_EXT_INIT_DONE
#define TENSIO_SQLITE_EXT_INIT_DONE
SQLITE_EXTENSION_INIT1
#endif

/* ================================================================
 * Virtual table base
 * ================================================================ */

typedef struct {
    sqlite3_vtab base;
    st_file_t   *model;
    const char  *prefix;
} safetensors_vtab_base;

typedef safetensors_vtab_base st_metadata_vtab;
typedef safetensors_vtab_base st_tensors_vtab;
typedef safetensors_vtab_base st_tensor_data_vtab;

typedef struct {
    sqlite3_vtab_cursor base;
    st_file_t          *model;
    int64_t             rowid;
    int                 eof;
} st_vtab_cursor;

int safetensors_register_vtables(sqlite3 *db, st_file_t *model,
                                 const char *prefix);

#endif /* SAFETENSORS_VTABLE_H */

/* ================================================================
 * IMPLEMENTATION
 * ================================================================ */

#ifdef SAFETENSORS_VTABLE_IMPLEMENTATION

/* ----------------------------------------------------------------
 * Shared helpers
 * ---------------------------------------------------------------- */

static int st_vtab_best_index(sqlite3_vtab *pVTab,
                              sqlite3_index_info *pIdxInfo) {
    (void)pVTab; (void)pIdxInfo;
    return SQLITE_OK;
}

static int st_vtab_disconnect(sqlite3_vtab *pVTab) {
    safetensors_vtab_base *p = (safetensors_vtab_base *)pVTab;
    if (p->prefix) sqlite3_free((void *)p->prefix);
    sqlite3_free(p);
    return SQLITE_OK;
}

static int st_vtab_readonly(sqlite3_vtab *pVTab, int argc,
                            sqlite3_value **argv,
                            sqlite3_int64 *pRowid) {
    (void)pVTab; (void)argc; (void)argv; (void)pRowid;
    return SQLITE_READONLY;
}

/* ================================================================
 * Metadata VTable (single-row: full header JSON)
 * ================================================================ */

static int st_metadata_connect(sqlite3 *db, void *pAux,
                               int argc, const char *const *argv,
                               sqlite3_vtab **ppVTab, char **pzErr) {
    st_metadata_vtab *p;
    (void)pzErr;
    p = (st_metadata_vtab *)sqlite3_malloc(sizeof(*p));
    if (!p) return SQLITE_NOMEM;
    memset(p, 0, sizeof(*p));
    p->model = (st_file_t *)pAux;
    p->prefix = (argc > 3 && argv[3])
        ? sqlite3_mprintf("%s", argv[3]) : NULL;

    *ppVTab = (sqlite3_vtab *)p;
    return sqlite3_declare_vtab(db,
        "CREATE TABLE x(metadata JSON, json_text TEXT)");
}

static int st_metadata_open(sqlite3_vtab *pVTab,
                            sqlite3_vtab_cursor **ppCursor) {
    st_vtab_cursor *c;
    (void)pVTab;
    c = (st_vtab_cursor *)sqlite3_malloc(sizeof(*c));
    if (!c) return SQLITE_NOMEM;
    memset(c, 0, sizeof(*c));
    *ppCursor = (sqlite3_vtab_cursor *)c;
    return SQLITE_OK;
}

static int st_metadata_close(sqlite3_vtab_cursor *pCursor) {
    sqlite3_free(pCursor);
    return SQLITE_OK;
}

static int st_metadata_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                              const char *idxStr, int argc,
                              sqlite3_value **argv) {
    st_vtab_cursor *c = (st_vtab_cursor *)pCursor;
    (void)idxNum; (void)idxStr; (void)argc; (void)argv;
    c->rowid = 0;
    c->eof = 0;
    return SQLITE_OK;
}

static int st_metadata_next(sqlite3_vtab_cursor *pCursor) {
    st_vtab_cursor *c = (st_vtab_cursor *)pCursor;
    c->eof = 1;
    return SQLITE_OK;
}

static int st_metadata_eof(sqlite3_vtab_cursor *pCursor) {
    return ((st_vtab_cursor *)pCursor)->eof;
}

static int st_metadata_column(sqlite3_vtab_cursor *pCursor,
                              sqlite3_context *pCtx, int col) {
    st_metadata_vtab *vt = (st_metadata_vtab *)pCursor->pVtab;
    if (col == 0 || col == 1) {
        sqlite3_result_text(pCtx, vt->model->header_json,
                            (int)vt->model->header_len, SQLITE_STATIC);
    }
    return SQLITE_OK;
}

static int st_metadata_rowid(sqlite3_vtab_cursor *pCursor,
                             sqlite3_int64 *pRowid) {
    *pRowid = 0;
    (void)pCursor;
    return SQLITE_OK;
}

/* ================================================================
 * Tensors VTable (one row per tensor)
 * ================================================================ */

static int st_tensors_connect(sqlite3 *db, void *pAux,
                              int argc, const char *const *argv,
                              sqlite3_vtab **ppVTab, char **pzErr) {
    st_tensors_vtab *p;
    (void)pzErr;
    p = (st_tensors_vtab *)sqlite3_malloc(sizeof(*p));
    if (!p) return SQLITE_NOMEM;
    memset(p, 0, sizeof(*p));
    p->model = (st_file_t *)pAux;
    p->prefix = (argc > 3 && argv[3])
        ? sqlite3_mprintf("%s", argv[3]) : NULL;

    *ppVTab = (sqlite3_vtab *)p;
    return sqlite3_declare_vtab(db,
        "CREATE TABLE x("
        "name TEXT PRIMARY KEY, dtype TEXT, shape TEXT, "
        "offset INTEGER, size INTEGER)");
}

static int st_tensors_open(sqlite3_vtab *pVTab,
                           sqlite3_vtab_cursor **ppCursor) {
    st_vtab_cursor *c;
    c = (st_vtab_cursor *)sqlite3_malloc(sizeof(*c));
    if (!c) return SQLITE_NOMEM;
    memset(c, 0, sizeof(*c));
    c->model = ((st_tensors_vtab *)pVTab)->model;
    *ppCursor = (sqlite3_vtab_cursor *)c;
    return SQLITE_OK;
}

static int st_tensors_close(sqlite3_vtab_cursor *pCursor) {
    sqlite3_free(pCursor);
    return SQLITE_OK;
}

static int st_tensors_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                             const char *idxStr, int argc,
                             sqlite3_value **argv) {
    st_vtab_cursor *c = (st_vtab_cursor *)pCursor;
    (void)idxNum; (void)idxStr; (void)argc; (void)argv;
    c->rowid = 0;
    c->eof = 0;
    return SQLITE_OK;
}

static int st_tensors_next(sqlite3_vtab_cursor *pCursor) {
    st_vtab_cursor *c = (st_vtab_cursor *)pCursor;
    c->rowid++;
    return SQLITE_OK;
}

static int st_tensors_eof(sqlite3_vtab_cursor *pCursor) {
    st_vtab_cursor *c = (st_vtab_cursor *)pCursor;
    return c->rowid >= c->model->num_tensors;
}

static const char *st_dtype_to_str(st_dtype_t dt) {
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
        default:      return "UNKNOWN";
    }
}

static int st_tensors_column(sqlite3_vtab_cursor *pCursor,
                             sqlite3_context *pCtx, int col) {
    st_vtab_cursor *c = (st_vtab_cursor *)pCursor;
    const st_tensor_t *t = &c->model->tensors[c->rowid];
    char buf[128];

    switch (col) {
        case 0: /* name */
            sqlite3_result_text(pCtx, t->name, -1, SQLITE_STATIC);
            break;
        case 1: /* dtype */
            sqlite3_result_text(pCtx, st_dtype_to_str(t->dtype), -1,
                                SQLITE_STATIC);
            break;
        case 2: { /* shape */
            int len = 0, i;
            len += snprintf(buf + len, sizeof(buf) - (size_t)len, "[");
            for (i = 0; i < 4 && t->shape[i] > 0; ++i) {
                if (i > 0)
                    len += snprintf(buf + len, sizeof(buf) - (size_t)len, ",");
                len += snprintf(buf + len, sizeof(buf) - (size_t)len,
                                "%llu", (unsigned long long)t->shape[i]);
            }
            snprintf(buf + len, sizeof(buf) - (size_t)len, "]");
            sqlite3_result_text(pCtx, buf, -1, SQLITE_TRANSIENT);
            break;
        }
        case 3: /* offset */
            sqlite3_result_int64(pCtx, (sqlite3_int64)t->offset);
            break;
        case 4: /* size */
            sqlite3_result_int64(pCtx, (sqlite3_int64)t->size);
            break;
    }
    return SQLITE_OK;
}

static int st_tensors_rowid(sqlite3_vtab_cursor *pCursor,
                            sqlite3_int64 *pRowid) {
    *pRowid = ((st_vtab_cursor *)pCursor)->rowid;
    return SQLITE_OK;
}

/* ================================================================
 * Tensor Data VTable (zero-copy BLOB)
 * ================================================================ */

static int st_data_connect(sqlite3 *db, void *pAux,
                           int argc, const char *const *argv,
                           sqlite3_vtab **ppVTab, char **pzErr) {
    st_tensor_data_vtab *p;
    (void)pzErr;
    p = (st_tensor_data_vtab *)sqlite3_malloc(sizeof(*p));
    if (!p) return SQLITE_NOMEM;
    memset(p, 0, sizeof(*p));
    p->model = (st_file_t *)pAux;
    p->prefix = (argc > 3 && argv[3])
        ? sqlite3_mprintf("%s", argv[3]) : NULL;

    *ppVTab = (sqlite3_vtab *)p;
    return sqlite3_declare_vtab(db,
        "CREATE TABLE x(tensor_name TEXT, data BLOB)");
}

static int st_data_open(sqlite3_vtab *pVTab,
                        sqlite3_vtab_cursor **ppCursor) {
    st_vtab_cursor *c;
    c = (st_vtab_cursor *)sqlite3_malloc(sizeof(*c));
    if (!c) return SQLITE_NOMEM;
    memset(c, 0, sizeof(*c));
    c->model = ((st_tensor_data_vtab *)pVTab)->model;
    *ppCursor = (sqlite3_vtab_cursor *)c;
    return SQLITE_OK;
}

static int st_data_close(sqlite3_vtab_cursor *pCursor) {
    sqlite3_free(pCursor);
    return SQLITE_OK;
}

static int st_data_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                          const char *idxStr, int argc,
                          sqlite3_value **argv) {
    st_vtab_cursor *c = (st_vtab_cursor *)pCursor;
    (void)idxNum; (void)idxStr; (void)argc; (void)argv;
    c->rowid = 0;
    c->eof = 0;
    return SQLITE_OK;
}

static int st_data_next(sqlite3_vtab_cursor *pCursor) {
    st_vtab_cursor *c = (st_vtab_cursor *)pCursor;
    c->rowid++;
    return SQLITE_OK;
}

static int st_data_eof(sqlite3_vtab_cursor *pCursor) {
    st_vtab_cursor *c = (st_vtab_cursor *)pCursor;
    return c->rowid >= c->model->num_tensors;
}

static int st_data_column(sqlite3_vtab_cursor *pCursor,
                          sqlite3_context *pCtx, int col) {
    st_vtab_cursor *c = (st_vtab_cursor *)pCursor;
    const st_tensor_t *t = &c->model->tensors[c->rowid];

    if (col == 0) {
        sqlite3_result_text(pCtx, t->name, -1, SQLITE_STATIC);
    } else if (col == 1) {
        const void *data = st_get_tensor_data(c->model, t);
        sqlite3_result_blob(pCtx, data, (int)t->size, SQLITE_STATIC);
    }
    return SQLITE_OK;
}

static int st_data_rowid(sqlite3_vtab_cursor *pCursor,
                         sqlite3_int64 *pRowid) {
    *pRowid = ((st_vtab_cursor *)pCursor)->rowid;
    return SQLITE_OK;
}

/* ================================================================
 * Module definitions
 * ================================================================ */

static sqlite3_module st_metadata_module = {
    .iVersion    = 0,
    .xConnect    = st_metadata_connect,
    .xBestIndex  = st_vtab_best_index,
    .xDisconnect = st_vtab_disconnect,
    .xOpen       = st_metadata_open,
    .xClose      = st_metadata_close,
    .xFilter     = st_metadata_filter,
    .xNext       = st_metadata_next,
    .xEof        = st_metadata_eof,
    .xColumn     = st_metadata_column,
    .xRowid      = st_metadata_rowid,
    .xUpdate     = st_vtab_readonly,
};

static sqlite3_module st_tensors_module = {
    .iVersion    = 0,
    .xConnect    = st_tensors_connect,
    .xBestIndex  = st_vtab_best_index,
    .xDisconnect = st_vtab_disconnect,
    .xOpen       = st_tensors_open,
    .xClose      = st_tensors_close,
    .xFilter     = st_tensors_filter,
    .xNext       = st_tensors_next,
    .xEof        = st_tensors_eof,
    .xColumn     = st_tensors_column,
    .xRowid      = st_tensors_rowid,
    .xUpdate     = st_vtab_readonly,
};

static sqlite3_module st_data_module = {
    .iVersion    = 0,
    .xConnect    = st_data_connect,
    .xBestIndex  = st_vtab_best_index,
    .xDisconnect = st_vtab_disconnect,
    .xOpen       = st_data_open,
    .xClose      = st_data_close,
    .xFilter     = st_data_filter,
    .xNext       = st_data_next,
    .xEof        = st_data_eof,
    .xColumn     = st_data_column,
    .xRowid      = st_data_rowid,
    .xUpdate     = st_vtab_readonly,
};

/* ================================================================
 * Registration
 * ================================================================ */

int safetensors_register_vtables(sqlite3 *db, st_file_t *model,
                                 const char *prefix) {
    int rc;
    char *meta_name, *tens_name, *data_name;

    if (prefix) {
        meta_name = sqlite3_mprintf("%s_metadata", prefix);
        tens_name = sqlite3_mprintf("%s_tensors", prefix);
        data_name = sqlite3_mprintf("%s_data", prefix);
    } else {
        meta_name = sqlite3_mprintf("safetensors_metadata");
        tens_name = sqlite3_mprintf("safetensors_tensors");
        data_name = sqlite3_mprintf("safetensors_data");
    }

    rc = sqlite3_create_module_v2(db, meta_name, &st_metadata_module,
                                  model, NULL);
    sqlite3_free(meta_name);
    if (rc != SQLITE_OK) { sqlite3_free(tens_name); sqlite3_free(data_name); return rc; }

    rc = sqlite3_create_module_v2(db, tens_name, &st_tensors_module,
                                  model, NULL);
    sqlite3_free(tens_name);
    if (rc != SQLITE_OK) { sqlite3_free(data_name); return rc; }

    rc = sqlite3_create_module_v2(db, data_name, &st_data_module,
                                  model, NULL);
    sqlite3_free(data_name);
    return rc;
}

#endif /* SAFETENSORS_VTABLE_IMPLEMENTATION */
