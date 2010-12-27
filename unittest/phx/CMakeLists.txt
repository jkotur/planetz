include_directories( ${PROJECT_SOURCE_DIR}/src/phx )
CUDA_ADD_LIBRARY( phx_test phx_test.cu )

target_link_libraries( phx_test phx util memmisc )
