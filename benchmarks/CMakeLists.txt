# --------------------------------------------------------------------------
# Tensio — Benchmarks
# --------------------------------------------------------------------------

# Macro: add a tensio benchmark executable
macro(tensio_add_benchmark BENCH_NAME BENCH_SOURCE)
    add_executable(${BENCH_NAME} ${BENCH_SOURCE})

    target_compile_definitions(${BENCH_NAME} PRIVATE
        SAFETENSORS_IMPLEMENTATION
        GGUF_IMPLEMENTATION
        TQ_IMPLEMENTATION
        CONVERT_IMPLEMENTATION
        _GNU_SOURCE
    )

    target_compile_options(${BENCH_NAME} PRIVATE -Werror)
    set_source_files_properties(${BENCH_SOURCE} PROPERTIES COMPILE_OPTIONS "-Wno-unused-function")
    target_link_libraries(${BENCH_NAME} PRIVATE tensio yyjson)
endmacro()

# Individual benchmark targets
tensio_add_benchmark(bench_parse   bench_parse.c)
tensio_add_benchmark(bench_dequant bench_dequant.c)
tensio_add_benchmark(bench_convert bench_convert.c)
