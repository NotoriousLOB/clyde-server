/* convert.c — Example: convert between tensor formats
 *
 * Usage: convert <input_file> <output_file>
 *
 * The output format is detected from the file extension:
 *   .safetensors → Safetensors
 *   .gguf        → GGUF
 *   .tq          → TQ
 */

#include "safetensors.h"
#include "gguf.h"
#include "tq.h"
#include "convert.h"
#include <stdio.h>

int main(int argc, char **argv) {
    int rc;

    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <input> <output>\n"
            "\n"
            "Supported extensions: .safetensors, .gguf, .tq\n"
            "Output format is inferred from the output file extension.\n",
            argv[0]);
        return 1;
    }

    printf("Converting: %s -> %s\n", argv[1], argv[2]);

    rc = convert_any_to_any(argv[1], argv[2]);
    if (rc != 0) {
        fprintf(stderr, "Error: conversion failed (rc=%d)\n", rc);
        return 1;
    }

    printf("Done.\n");
    return 0;
}
