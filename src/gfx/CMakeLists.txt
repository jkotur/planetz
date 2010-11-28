include_directories( . )

add_library( gfx arrow background gfx texture
	planetz_renderer shader
	planetz_picker
	deffered_renderer )
target_link_libraries( gfx ${OPENGL_LIBRARY} ${SDL_LIBRARY} ${SDLIMAGE_LIBRARY} )
