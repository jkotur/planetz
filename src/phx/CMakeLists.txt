include_directories( . )

CUDA_ADD_LIBRARY( phx phx.cu phx_kernels.cu )

target_link_libraries( phx )
