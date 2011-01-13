include_directories( ${PROJECT_SOURCE_DIR}/src/ui )
add_library( ui_test planetz_picker_test )

target_link_libraries( ui_test ui )
