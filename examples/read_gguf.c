/* read_gguf.c — Example: parse and list metadata + tensors in a .gguf file
 *
 * Usage: read_gguf <file.gguf>
 */

#include "gguf.h"
#include <stdio.h>

static const char *meta_type_str(gguf_meta_type_t t) {
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
        default: return "???";
    }
}

static const char *tensor_type_str(gguf_type_t t) {
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
        default: return "???";
    }
}

int main(int argc, char **argv) {
    gguf_mmap_t mm;
    gguf_file_t f;
    uint64_t i;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file.gguf>\n", argv[0]);
        return 1;
    }

    if (gguf_mmap(argv[1], &mm) != 0) {
        fprintf(stderr, "Error: cannot open '%s'\n", argv[1]);
        return 1;
    }

    if (gguf_parse(&mm, &f) != 0) {
        fprintf(stderr, "Error: cannot parse '%s'\n", argv[1]);
        gguf_munmap(&mm);
        return 1;
    }

    printf("File:      %s\n", argv[1]);
    printf("Version:   %u\n", f.version);
    printf("Metadata:  %llu entries\n", (unsigned long long)f.metadata_count);
    printf("Tensors:   %llu\n\n", (unsigned long long)f.tensor_count);

    /* Metadata */
    if (f.metadata_count > 0) {
        printf("--- Metadata ---\n");
        printf("%-40s  %-10s  %s\n", "KEY", "TYPE", "VALUE");
        for (i = 0; i < f.metadata_count; ++i) {
            const gguf_kv_t *kv = &f.metadata[i];
            printf("%-40s  %-10s  ", kv->key, meta_type_str(kv->type));
            switch (kv->type) {
                case GGUF_META_STRING:
                    printf("%.*s", (int)kv->value.str.len, kv->value.str.data);
                    break;
                case GGUF_META_UINT32: printf("%u", kv->value.u32); break;
                case GGUF_META_INT32:  printf("%d", kv->value.i32); break;
                case GGUF_META_FLOAT32:printf("%g", (double)kv->value.f32); break;
                case GGUF_META_BOOL:   printf("%s", kv->value.b ? "true" : "false"); break;
                case GGUF_META_UINT64: printf("%llu", (unsigned long long)kv->value.u64); break;
                case GGUF_META_INT64:  printf("%lld", (long long)kv->value.i64); break;
                default: printf("(...)"); break;
            }
            printf("\n");
        }
        printf("\n");
    }

    /* Tensors */
    printf("--- Tensors ---\n");
    printf("%-50s  %-6s  %-10s  %s\n", "NAME", "TYPE", "SIZE", "SHAPE");
    for (i = 0; i < f.tensor_count; ++i) {
        const gguf_tensor_t *t = &f.tensors[i];
        char shape[128];
        int len = 0;
        uint32_t d;

        len += snprintf(shape + len, sizeof(shape) - (size_t)len, "[");
        for (d = 0; d < t->n_dims; ++d) {
            if (d > 0) len += snprintf(shape + len, sizeof(shape) - (size_t)len, ", ");
            len += snprintf(shape + len, sizeof(shape) - (size_t)len,
                            "%llu", (unsigned long long)t->ne[d]);
        }
        snprintf(shape + len, sizeof(shape) - (size_t)len, "]");

        printf("%-50s  %-6s  %-10llu  %s\n",
               t->name, tensor_type_str(t->type),
               (unsigned long long)t->size, shape);
    }

    gguf_free(&f);
    gguf_munmap(&mm);
    return 0;
}
