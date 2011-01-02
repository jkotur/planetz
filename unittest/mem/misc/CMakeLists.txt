include_directories( ${PROJECT_SOURCE_DIR}/src/mem/misc )
CUDA_ADD_LIBRARY( memmisc_test buffer_test )

target_link_libraries( memmisc_test )
