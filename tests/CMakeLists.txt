# --------------------------------------------------------------------------
# Tensio — Test Suite
# --------------------------------------------------------------------------

# Macro: add a tensio test executable
# Each test gets the *_IMPLEMENTATION defines so it can exercise the full API.
macro(tensio_add_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})

    target_compile_definitions(${TEST_NAME} PRIVATE
        SAFETENSORS_IMPLEMENTATION
        GGUF_IMPLEMENTATION
        TQ_IMPLEMENTATION
        CONVERT_IMPLEMENTATION
    )

    target_compile_options(${TEST_NAME} PRIVATE -Werror)
    set_source_files_properties(${TEST_SOURCE} PROPERTIES COMPILE_OPTIONS "-Wno-unused-function")
    target_link_libraries(${TEST_NAME} PRIVATE tensio yyjson)

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endmacro()

# Individual test targets
tensio_add_test(test_safetensors test_safetensors.c)
tensio_add_test(test_gguf        test_gguf.c)
tensio_add_test(test_tq          test_tq.c)
tensio_add_test(test_convert     test_convert.c)
