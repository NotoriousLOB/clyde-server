/* tensio CLI — minimal entry point
 * Full CLI implemented in Phase 7
 */

/* Implementation macros are set via CMake target_compile_definitions.
 * For standalone compilation, define them on the command line:
 *   -DSAFETENSORS_IMPLEMENTATION -DGGUF_IMPLEMENTATION ...
 */

#include "tensio.h"

#include <stdio.h>
#include <string.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <command> [args...]\n"
        "\n"
        "Commands:\n"
        "  info    <file>          Show file format and tensor summary\n"
        "  list    <file>          List all tensors\n"
        "  convert <in> <out>      Convert between formats\n"
        "\n"
        "Supported formats: .safetensors, .gguf, .tq\n",
        prog);
}

static int cmd_info(const char *path) {
    /* Try GGUF first (has magic) */
    {
        gguf_mmap_t mm;
        if (gguf_mmap(path, &mm) == 0) {
            gguf_file_t f;
            if (gguf_parse(&mm, &f) == 0) {
                printf("Format:    GGUF v%u\n", f.version);
                printf("Tensors:   %llu\n", (unsigned long long)f.tensor_count);
                printf("Metadata:  %llu\n", (unsigned long long)f.metadata_count);
                gguf_free(&f);
                gguf_munmap(&mm);
                return 0;
            }
            gguf_munmap(&mm);
        }
    }

    /* Try TQ */
    {
        tq_file_t f;
        if (tq_mmap(path, &f) == 0) {
            printf("Format:    TQ v%u\n", f.hdr->version);
            printf("Tensors:   %llu\n", (unsigned long long)f.hdr->tensor_count);
            printf("Family:    %u\n", f.hdr->model_family_id);
            printf("Features:  0x%llx\n", (unsigned long long)f.hdr->features);
            tq_munmap(&f);
            return 0;
        }
    }

    /* Try Safetensors (no magic, assume if others fail) */
    {
        st_mmap_t mm;
        if (st_mmap(path, &mm) == 0) {
            st_file_t f;
            if (st_parse(&mm, &f) == 0) {
                printf("Format:    Safetensors\n");
                printf("Tensors:   %u\n", f.num_tensors);
                printf("Header:    %llu bytes\n",
                       (unsigned long long)f.header_len);
                printf("Data:      %llu bytes\n",
                       (unsigned long long)f.data_size);
                st_free(&f);
                st_munmap(&mm);
                return 0;
            }
            st_munmap(&mm);
        }
    }

    fprintf(stderr, "Error: could not open or parse '%s'\n", path);
    return 1;
}

static int cmd_list(const char *path) {
    /* Try GGUF */
    {
        gguf_mmap_t mm;
        if (gguf_mmap(path, &mm) == 0) {
            gguf_file_t f;
            if (gguf_parse(&mm, &f) == 0) {
                uint64_t i;
                printf("%-60s  %-8s  %s\n", "NAME", "TYPE", "SIZE");
                for (i = 0; i < f.tensor_count; ++i) {
                    printf("%-60s  %-8u  %llu\n",
                           f.tensors[i].name,
                           (unsigned)f.tensors[i].type,
                           (unsigned long long)f.tensors[i].size);
                }
                gguf_free(&f);
                gguf_munmap(&mm);
                return 0;
            }
            gguf_munmap(&mm);
        }
    }

    /* Try TQ */
    {
        tq_file_t f;
        if (tq_mmap(path, &f) == 0) {
            uint64_t i;
            printf("%-60s  %s  %s  %s\n", "NAME", "B", "ROWS", "COLS");
            for (i = 0; i < f.hdr->tensor_count; ++i) {
                printf("%-60s  %u  %u  %u\n",
                       f.tensors[i].name,
                       f.tensors[i].b,
                       f.tensors[i].rows,
                       f.tensors[i].cols);
            }
            tq_munmap(&f);
            return 0;
        }
    }

    /* Try Safetensors */
    {
        st_mmap_t mm;
        if (st_mmap(path, &mm) == 0) {
            st_file_t f;
            if (st_parse(&mm, &f) == 0) {
                uint32_t i;
                printf("%-60s  %-6s  %s\n", "NAME", "DTYPE", "SIZE");
                for (i = 0; i < f.num_tensors; ++i) {
                    printf("%-60s  %-6u  %llu\n",
                           f.tensors[i].name,
                           (unsigned)f.tensors[i].dtype,
                           (unsigned long long)f.tensors[i].size);
                }
                st_free(&f);
                st_munmap(&mm);
                return 0;
            }
            st_munmap(&mm);
        }
    }

    fprintf(stderr, "Error: could not open or parse '%s'\n", path);
    return 1;
}

static int cmd_convert(const char *in, const char *out) {
    int rc = convert_any_to_any(in, out);
    if (rc != 0) {
        fprintf(stderr, "Error: conversion failed\n");
        return 1;
    }
    printf("Converted %s -> %s\n", in, out);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "info") == 0) {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        return cmd_info(argv[2]);
    }
    if (strcmp(argv[1], "list") == 0) {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        return cmd_list(argv[2]);
    }
    if (strcmp(argv[1], "convert") == 0) {
        if (argc < 4) { print_usage(argv[0]); return 1; }
        return cmd_convert(argv[2], argv[3]);
    }

    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    print_usage(argv[0]);
    return 1;
}
