include_directories( . )

add_library( gfx arrow background planet planet_gfx gfx texture planetz_renderer )
target_link_libraries( gfx ${OPENGL_LIBRARY} ${SDL_LIBRARY} ${SDLIMAGE_LIBRARY} )
