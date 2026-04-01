/* read_safetensors.c — Example: parse and list tensors in a .safetensors file
 *
 * Usage: read_safetensors <file.safetensors>
 */

#include "safetensors.h"
#include <stdio.h>

static const char *dtype_str(st_dtype_t dt) {
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
        default:      return "???";
    }
}

int main(int argc, char **argv) {
    st_mmap_t mm;
    st_file_t f;
    uint32_t i;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file.safetensors>\n", argv[0]);
        return 1;
    }

    if (st_mmap(argv[1], &mm) != 0) {
        fprintf(stderr, "Error: cannot open '%s'\n", argv[1]);
        return 1;
    }

    if (st_parse(&mm, &f) != 0) {
        fprintf(stderr, "Error: cannot parse '%s'\n", argv[1]);
        st_munmap(&mm);
        return 1;
    }

    printf("File:      %s\n", argv[1]);
    printf("Header:    %llu bytes JSON\n", (unsigned long long)f.header_len);
    printf("Data:      %llu bytes\n", (unsigned long long)f.data_size);
    printf("Tensors:   %u\n\n", f.num_tensors);

    printf("%-50s  %-6s  %-10s  %s\n", "NAME", "DTYPE", "SIZE", "SHAPE");
    printf("%-50s  %-6s  %-10s  %s\n",
           "--------------------------------------------------",
           "------", "----------", "----------");

    for (i = 0; i < f.num_tensors; ++i) {
        const st_tensor_t *t = &f.tensors[i];
        char shape[128];
        int len = 0;
        uint32_t d;

        len += snprintf(shape + len, sizeof(shape) - (size_t)len, "[");
        for (d = 0; d < t->ndim; ++d) {
            if (d > 0) len += snprintf(shape + len, sizeof(shape) - (size_t)len, ", ");
            len += snprintf(shape + len, sizeof(shape) - (size_t)len,
                            "%llu", (unsigned long long)t->shape[d]);
        }
        snprintf(shape + len, sizeof(shape) - (size_t)len, "]");

        printf("%-50s  %-6s  %-10llu  %s\n",
               t->name, dtype_str(t->dtype),
               (unsigned long long)t->size, shape);
    }

    st_free(&f);
    st_munmap(&mm);
    return 0;
}
