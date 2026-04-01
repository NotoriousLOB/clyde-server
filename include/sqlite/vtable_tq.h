/* vtable_tq.h — Full SQLite3 Virtual Table for .tq files
 * Strict C99, zero UB, header-only, zero-copy BLOBs, table prefix support
 * Re-uses tq.h (include before this file)
 */

#ifndef TQ_VTABLE_H
#define TQ_VTABLE_H

#include "tq.h"
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
    tq_file_t   *model;
    const char  *prefix;
} tq_vtab_base;

typedef tq_vtab_base tq_metadata_vtab;
typedef tq_vtab_base tq_tensors_vtab;
typedef tq_vtab_base tq_data_vtab;

typedef struct {
    sqlite3_vtab_cursor base;
    tq_file_t          *model;
    int64_t             rowid;
    int                 eof;
} tq_vtab_cursor;

int tq_register_vtables(sqlite3 *db, tq_file_t *model, const char *prefix);

#endif /* TQ_VTABLE_H */

/* ================================================================
 * IMPLEMENTATION (define TQ_VTABLE_IMPLEMENTATION in one .c)
 * ================================================================ */

#ifdef TQ_VTABLE_IMPLEMENTATION

/* ----------------------------------------------------------------
 * Shared helpers
 * ---------------------------------------------------------------- */

static int tq_vtab_best_index(sqlite3_vtab *pVTab,
                              sqlite3_index_info *pIdxInfo) {
    (void)pVTab; (void)pIdxInfo;
    return SQLITE_OK;
}

static int tq_vtab_disconnect(sqlite3_vtab *pVTab) {
    tq_vtab_base *p = (tq_vtab_base *)pVTab;
    if (p->prefix) sqlite3_free((void *)p->prefix);
    sqlite3_free(p);
    return SQLITE_OK;
}

static int tq_vtab_readonly(sqlite3_vtab *pVTab, int argc,
                            sqlite3_value **argv,
                            sqlite3_int64 *pRowid) {
    (void)pVTab; (void)argc; (void)argv; (void)pRowid;
    return SQLITE_READONLY;
}

/* ================================================================
 * Metadata VTable (single-row: header fields as JSON)
 * ================================================================ */

static int tq_meta_connect(sqlite3 *db, void *pAux,
                           int argc, const char *const *argv,
                           sqlite3_vtab **ppVTab, char **pzErr) {
    tq_metadata_vtab *p;
    (void)pzErr;
    p = (tq_metadata_vtab *)sqlite3_malloc(sizeof(*p));
    if (!p) return SQLITE_NOMEM;
    memset(p, 0, sizeof(*p));
    p->model = (tq_file_t *)pAux;
    p->prefix = (argc > 3 && argv[3])
        ? sqlite3_mprintf("%s", argv[3]) : NULL;

    *ppVTab = (sqlite3_vtab *)p;
    return sqlite3_declare_vtab(db,
        "CREATE TABLE x("
        "magic INTEGER, version INTEGER, features INTEGER, "
        "tensor_count INTEGER, data_offset INTEGER, "
        "total_data_size INTEGER, model_family_id INTEGER, "
        "model_version INTEGER)");
}

static int tq_meta_open(sqlite3_vtab *pVTab,
                        sqlite3_vtab_cursor **ppCursor) {
    tq_vtab_cursor *c;
    (void)pVTab;
    c = (tq_vtab_cursor *)sqlite3_malloc(sizeof(*c));
    if (!c) return SQLITE_NOMEM;
    memset(c, 0, sizeof(*c));
    *ppCursor = (sqlite3_vtab_cursor *)c;
    return SQLITE_OK;
}

static int tq_meta_close(sqlite3_vtab_cursor *pCursor) {
    sqlite3_free(pCursor);
    return SQLITE_OK;
}

static int tq_meta_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                          const char *idxStr, int argc,
                          sqlite3_value **argv) {
    tq_vtab_cursor *c = (tq_vtab_cursor *)pCursor;
    (void)idxNum; (void)idxStr; (void)argc; (void)argv;
    c->rowid = 0;
    c->eof = 0;
    return SQLITE_OK;
}

static int tq_meta_next(sqlite3_vtab_cursor *pCursor) {
    ((tq_vtab_cursor *)pCursor)->eof = 1;
    return SQLITE_OK;
}

static int tq_meta_eof(sqlite3_vtab_cursor *pCursor) {
    return ((tq_vtab_cursor *)pCursor)->eof;
}

static int tq_meta_column(sqlite3_vtab_cursor *pCursor,
                          sqlite3_context *pCtx, int col) {
    tq_metadata_vtab *vt = (tq_metadata_vtab *)pCursor->pVtab;
    const tq_header_t *h = vt->model->hdr;

    switch (col) {
        case 0: sqlite3_result_int(pCtx, (int)h->magic); break;
        case 1: sqlite3_result_int(pCtx, (int)h->version); break;
        case 2: sqlite3_result_int64(pCtx, (sqlite3_int64)h->features); break;
        case 3: sqlite3_result_int64(pCtx, (sqlite3_int64)h->tensor_count); break;
        case 4: sqlite3_result_int64(pCtx, (sqlite3_int64)h->data_offset); break;
        case 5: sqlite3_result_int64(pCtx, (sqlite3_int64)h->total_data_size); break;
        case 6: sqlite3_result_int(pCtx, (int)h->model_family_id); break;
        case 7: sqlite3_result_int(pCtx, (int)h->model_version); break;
    }
    return SQLITE_OK;
}

static int tq_meta_rowid(sqlite3_vtab_cursor *pCursor,
                         sqlite3_int64 *pRowid) {
    *pRowid = 0;
    (void)pCursor;
    return SQLITE_OK;
}

/* ================================================================
 * Tensors VTable (one row per tensor descriptor)
 * ================================================================ */

static int tq_tensors_connect(sqlite3 *db, void *pAux,
                              int argc, const char *const *argv,
                              sqlite3_vtab **ppVTab, char **pzErr) {
    tq_tensors_vtab *p;
    (void)pzErr;
    p = (tq_tensors_vtab *)sqlite3_malloc(sizeof(*p));
    if (!p) return SQLITE_NOMEM;
    memset(p, 0, sizeof(*p));
    p->model = (tq_file_t *)pAux;
    p->prefix = (argc > 3 && argv[3])
        ? sqlite3_mprintf("%s", argv[3]) : NULL;

    *ppVTab = (sqlite3_vtab *)p;
    return sqlite3_declare_vtab(db,
        "CREATE TABLE x("
        "name TEXT PRIMARY KEY, b INTEGER, "
        "rows INTEGER, cols INTEGER, "
        "frame_offset INTEGER, frame_size INTEGER, "
        "unpacked_size INTEGER)");
}

static int tq_tensors_open(sqlite3_vtab *pVTab,
                           sqlite3_vtab_cursor **ppCursor) {
    tq_vtab_cursor *c;
    c = (tq_vtab_cursor *)sqlite3_malloc(sizeof(*c));
    if (!c) return SQLITE_NOMEM;
    memset(c, 0, sizeof(*c));
    c->model = ((tq_tensors_vtab *)pVTab)->model;
    *ppCursor = (sqlite3_vtab_cursor *)c;
    return SQLITE_OK;
}

static int tq_tensors_close(sqlite3_vtab_cursor *pCursor) {
    sqlite3_free(pCursor);
    return SQLITE_OK;
}

static int tq_tensors_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                             const char *idxStr, int argc,
                             sqlite3_value **argv) {
    tq_vtab_cursor *c = (tq_vtab_cursor *)pCursor;
    (void)idxNum; (void)idxStr; (void)argc; (void)argv;
    c->rowid = 0;
    c->eof = (c->model->hdr->tensor_count == 0) ? 1 : 0;
    return SQLITE_OK;
}

static int tq_tensors_next(sqlite3_vtab_cursor *pCursor) {
    tq_vtab_cursor *c = (tq_vtab_cursor *)pCursor;
    c->rowid++;
    if ((uint64_t)c->rowid >= c->model->hdr->tensor_count)
        c->eof = 1;
    return SQLITE_OK;
}

static int tq_tensors_eof(sqlite3_vtab_cursor *pCursor) {
    return ((tq_vtab_cursor *)pCursor)->eof;
}

static int tq_tensors_column(sqlite3_vtab_cursor *pCursor,
                             sqlite3_context *pCtx, int col) {
    tq_vtab_cursor *c = (tq_vtab_cursor *)pCursor;
    const tq_tensor_t *t = &c->model->tensors[c->rowid];

    switch (col) {
        case 0: sqlite3_result_text(pCtx, t->name, -1, SQLITE_STATIC); break;
        case 1: sqlite3_result_int(pCtx, (int)t->b); break;
        case 2: sqlite3_result_int(pCtx, (int)t->rows); break;
        case 3: sqlite3_result_int(pCtx, (int)t->cols); break;
        case 4: sqlite3_result_int64(pCtx, (sqlite3_int64)t->frame_offset); break;
        case 5: sqlite3_result_int64(pCtx, (sqlite3_int64)t->frame_size); break;
        case 6: sqlite3_result_int64(pCtx, (sqlite3_int64)t->unpacked_size); break;
    }
    return SQLITE_OK;
}

static int tq_tensors_rowid(sqlite3_vtab_cursor *pCursor,
                            sqlite3_int64 *pRowid) {
    *pRowid = ((tq_vtab_cursor *)pCursor)->rowid;
    return SQLITE_OK;
}

/* ================================================================
 * Data VTable (zero-copy BLOB)
 * ================================================================ */

static int tq_data_connect(sqlite3 *db, void *pAux,
                           int argc, const char *const *argv,
                           sqlite3_vtab **ppVTab, char **pzErr) {
    tq_data_vtab *p;
    (void)pzErr;
    p = (tq_data_vtab *)sqlite3_malloc(sizeof(*p));
    if (!p) return SQLITE_NOMEM;
    memset(p, 0, sizeof(*p));
    p->model = (tq_file_t *)pAux;
    p->prefix = (argc > 3 && argv[3])
        ? sqlite3_mprintf("%s", argv[3]) : NULL;

    *ppVTab = (sqlite3_vtab *)p;
    return sqlite3_declare_vtab(db,
        "CREATE TABLE x(tensor_name TEXT, data BLOB)");
}

static int tq_data_open(sqlite3_vtab *pVTab,
                        sqlite3_vtab_cursor **ppCursor) {
    tq_vtab_cursor *c;
    c = (tq_vtab_cursor *)sqlite3_malloc(sizeof(*c));
    if (!c) return SQLITE_NOMEM;
    memset(c, 0, sizeof(*c));
    c->model = ((tq_data_vtab *)pVTab)->model;
    *ppCursor = (sqlite3_vtab_cursor *)c;
    return SQLITE_OK;
}

static int tq_data_close(sqlite3_vtab_cursor *pCursor) {
    sqlite3_free(pCursor);
    return SQLITE_OK;
}

static int tq_data_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                          const char *idxStr, int argc,
                          sqlite3_value **argv) {
    tq_vtab_cursor *c = (tq_vtab_cursor *)pCursor;
    (void)idxNum; (void)idxStr; (void)argc; (void)argv;
    c->rowid = 0;
    c->eof = (c->model->hdr->tensor_count == 0) ? 1 : 0;
    return SQLITE_OK;
}

static int tq_data_next(sqlite3_vtab_cursor *pCursor) {
    tq_vtab_cursor *c = (tq_vtab_cursor *)pCursor;
    c->rowid++;
    if ((uint64_t)c->rowid >= c->model->hdr->tensor_count)
        c->eof = 1;
    return SQLITE_OK;
}

static int tq_data_eof(sqlite3_vtab_cursor *pCursor) {
    return ((tq_vtab_cursor *)pCursor)->eof;
}

static int tq_data_column(sqlite3_vtab_cursor *pCursor,
                          sqlite3_context *pCtx, int col) {
    tq_vtab_cursor *c = (tq_vtab_cursor *)pCursor;
    const tq_tensor_t *t = &c->model->tensors[c->rowid];

    if (col == 0) {
        sqlite3_result_text(pCtx, t->name, -1, SQLITE_STATIC);
    } else if (col == 1) {
        const void *data = tq_get_tensor_data(c->model, t);
        sqlite3_result_blob(pCtx, data, (int)t->unpacked_size,
                            SQLITE_STATIC);
    }
    return SQLITE_OK;
}

static int tq_data_rowid(sqlite3_vtab_cursor *pCursor,
                         sqlite3_int64 *pRowid) {
    *pRowid = ((tq_vtab_cursor *)pCursor)->rowid;
    return SQLITE_OK;
}

/* ================================================================
 * Module definitions
 * ================================================================ */

static sqlite3_module tq_metadata_module = {
    .iVersion    = 0,
    .xConnect    = tq_meta_connect,
    .xBestIndex  = tq_vtab_best_index,
    .xDisconnect = tq_vtab_disconnect,
    .xOpen       = tq_meta_open,
    .xClose      = tq_meta_close,
    .xFilter     = tq_meta_filter,
    .xNext       = tq_meta_next,
    .xEof        = tq_meta_eof,
    .xColumn     = tq_meta_column,
    .xRowid      = tq_meta_rowid,
    .xUpdate     = tq_vtab_readonly,
};

static sqlite3_module tq_tensors_module = {
    .iVersion    = 0,
    .xConnect    = tq_tensors_connect,
    .xBestIndex  = tq_vtab_best_index,
    .xDisconnect = tq_vtab_disconnect,
    .xOpen       = tq_tensors_open,
    .xClose      = tq_tensors_close,
    .xFilter     = tq_tensors_filter,
    .xNext       = tq_tensors_next,
    .xEof        = tq_tensors_eof,
    .xColumn     = tq_tensors_column,
    .xRowid      = tq_tensors_rowid,
    .xUpdate     = tq_vtab_readonly,
};

static sqlite3_module tq_data_module = {
    .iVersion    = 0,
    .xConnect    = tq_data_connect,
    .xBestIndex  = tq_vtab_best_index,
    .xDisconnect = tq_vtab_disconnect,
    .xOpen       = tq_data_open,
    .xClose      = tq_data_close,
    .xFilter     = tq_data_filter,
    .xNext       = tq_data_next,
    .xEof        = tq_data_eof,
    .xColumn     = tq_data_column,
    .xRowid      = tq_data_rowid,
    .xUpdate     = tq_vtab_readonly,
};

/* ================================================================
 * Registration
 * ================================================================ */

int tq_register_vtables(sqlite3 *db, tq_file_t *model, const char *prefix) {
    int rc;
    char *meta_name, *tens_name, *data_name;

    if (prefix) {
        meta_name = sqlite3_mprintf("%s_metadata", prefix);
        tens_name = sqlite3_mprintf("%s_tensors", prefix);
        data_name = sqlite3_mprintf("%s_data", prefix);
    } else {
        meta_name = sqlite3_mprintf("tq_metadata");
        tens_name = sqlite3_mprintf("tq_tensors");
        data_name = sqlite3_mprintf("tq_data");
    }

    rc = sqlite3_create_module_v2(db, meta_name, &tq_metadata_module,
                                  model, NULL);
    sqlite3_free(meta_name);
    if (rc != SQLITE_OK) { sqlite3_free(tens_name); sqlite3_free(data_name); return rc; }

    rc = sqlite3_create_module_v2(db, tens_name, &tq_tensors_module,
                                  model, NULL);
    sqlite3_free(tens_name);
    if (rc != SQLITE_OK) { sqlite3_free(data_name); return rc; }

    rc = sqlite3_create_module_v2(db, data_name, &tq_data_module,
                                  model, NULL);
    sqlite3_free(data_name);
    return rc;
}

#endif /* TQ_VTABLE_IMPLEMENTATION */
