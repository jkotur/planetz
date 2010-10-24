include_directories(
	${PROJECT_SOURCE_DIR}/src 
	${PROJECT_SOURCE_DIR}/src/ui )

add_library( input driver input )
target_link_libraries( input ${SDL_LIBRARY} )
