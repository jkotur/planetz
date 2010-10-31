include_directories( . )

add_library( gpu gfx_planet_factory phx_planet_factory holder )

target_link_libraries( gpu ${OPENGL_LIBRARY} ${CUDA_LIBRARIES} )

