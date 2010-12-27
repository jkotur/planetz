cmake_policy(VERSION 2.6)

set( INCLUDE_DIRS
	.
	${PROJECT_INCLUDE_DIR}
	${CUDA_INCLUDE_DIRS}
	${OPENGL_INCLUDE_DIR}
	${SDL_INCLUDE_DIR}
	${CEGUI_INCLUDE_DIR}
	${Boost_INCLUDE_DIR} )

include_directories( ${INCLUDE_DIRS} )

add_subdirectory( 3rdparty )
add_subdirectory( debug )
add_subdirectory( gfx )
add_subdirectory( mem )
add_subdirectory( phx )
add_subdirectory( ui )
add_subdirectory( util )

add_executable( inz application constants main window options )

target_link_libraries( inz 
	dbg gfx mem phx ui util # project libs
	cudpp cutil # third party libs
	${CUDA_LIBRARIES}
	${OPENGL_LIBRARY}
	${SDL_LIBRARY}
	${SDLIMAGE_LIBRARY}
	${CEGUI_LIBRARIES}
	${CEGUIOPENGL_LIBRARIES}
	${Boost_LIBRARIES} 
	${SQLITE3_LIBRARIES} )

