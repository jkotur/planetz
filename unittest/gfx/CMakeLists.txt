include_directories( ${PROJECT_SOURCE_DIR}/src/gfx )
add_library( gfx_test shader_test planetz_picker_test )

target_link_libraries( gfx_test gfx )
