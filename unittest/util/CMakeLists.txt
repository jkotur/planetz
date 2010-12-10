include_directories( ${PROJECT_SOURCE_DIR}/src/util )
add_library( util_test config_test )

target_link_libraries( util_test )
