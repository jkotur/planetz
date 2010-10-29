cmake_policy(VERSION 2.6)

set( INCLUDE_DIRS
	.
	${CUDA_INCLUDE_DIRS}
	${OPENGL_INCLUDE_DIR}
	${SDL_INCLUDE_DIR}
	${CEGUI_INCLUDE_DIR}
	${Boost_INCLUDE_DIR} )

include_directories( ${INCLUDE_DIRS} )

add_subdirectory( gfx )
add_subdirectory( gpu )
add_subdirectory( mem )
add_subdirectory( phx )
add_subdirectory( sphere )
add_subdirectory( ui )
add_subdirectory( util )

add_executable( inz application constants planet planetz_manager main )

target_link_libraries( inz gfx gpu mem phx sphere ui util 
	${CUDA_LIBRARIES}
	${OPENGL_LIBRARY}
	${SDL_LIBRARY}
	${SDLIMAGE_LIBRARY}
	${CEGUI_LIBRARIES}
	${CEGUIOPENGL_LIBRARIES}
	${Boost_LIBRARIES} )

