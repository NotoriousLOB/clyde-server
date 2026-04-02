# --------------------------------------------------------------------------
# Tensio — Examples
# --------------------------------------------------------------------------

# Macro: add a tensio example executable
macro(tensio_add_example EXAMPLE_NAME EXAMPLE_SOURCE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})

    target_compile_definitions(${EXAMPLE_NAME} PRIVATE
        SAFETENSORS_IMPLEMENTATION
        GGUF_IMPLEMENTATION
        TQ_IMPLEMENTATION
        CONVERT_IMPLEMENTATION
        _GNU_SOURCE
    )

    target_compile_options(${EXAMPLE_NAME} PRIVATE -Werror)
    set_source_files_properties(${EXAMPLE_SOURCE} PROPERTIES COMPILE_OPTIONS "-Wno-unused-function")
    target_link_libraries(${EXAMPLE_NAME} PRIVATE tensio yyjson)
endmacro()

# Individual example targets
tensio_add_example(read_safetensors read_safetensors.c)
tensio_add_example(read_gguf        read_gguf.c)
tensio_add_example(convert          convert.c)
