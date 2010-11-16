include_directories( . )

add_library( gfx arrow background gfx texture
	planetz_renderer shader
	geometry_renderer copy_renderer
	deffered_renderer )
target_link_libraries( gfx ${OPENGL_LIBRARY} ${SDL_LIBRARY} ${SDLIMAGE_LIBRARY} )
