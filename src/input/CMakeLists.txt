include_directories( . )

add_library( input driver input )
target_link_libraries( input ${SDL_LIBRARY} )
