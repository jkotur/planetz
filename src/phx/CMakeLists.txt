include_directories( . )

CUDA_ADD_LIBRARY( phx phx.cu phx_kernels.cu kmeans.cu kmeans_kernel.cu )

target_link_libraries( phx cudpp )
