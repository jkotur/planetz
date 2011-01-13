include_directories( . )

add_library( gfx background gfx texture
	shader
	planetz_tracer
	deffered_renderer )
target_link_libraries( gfx ${OPENGL_LIBRARY} ${SDL_LIBRARY} ${SDLIMAGE_LIBRARY} )
