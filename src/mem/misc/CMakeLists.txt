include_directories( . ../../3rdparty/cudpp/include/
	../../3rdparty/cudpp/src/ )

CUDA_ADD_LIBRARY( memmisc gfx_planet_factory phx_planet_factory holder buffer
	buffer_cu buffer_cpu materials_manager saver_params holder_kernels.cu
	holder_kernels.h holder_cleaner.cu holder_cleaner_kernels.cu )

target_link_libraries( memmisc ${OPENGL_LIBRARY} ${CUDA_LIBRARIES} )

