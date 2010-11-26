include_directories( . )

add_library( memmisc gfx_planet_factory phx_planet_factory holder buffer
	buffer_cu buffer_cpu materials_manager )

target_link_libraries( memmisc ${OPENGL_LIBRARY} ${CUDA_LIBRARIES} )

