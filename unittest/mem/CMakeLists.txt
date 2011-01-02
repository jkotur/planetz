include_directories( ${PROJECT_SOURCE_DIR}/src/mem )

#add_subdirectory( db )
add_subdirectory( misc )

CUDA_ADD_LIBRARY( mem_test )

target_link_libraries( mem_test memmisc db )
