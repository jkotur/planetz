
find_package( CppUnit )

set( INCLUDE_DIRS
	.
	${PROJECT_SOURCE_DIR}/src
	${CUDA_INCLUDE_DIRS}
	${OPENGL_INCLUDE_DIR}
	${SDL_INCLUDE_DIR}
	${CEGUI_INCLUDE_DIR}
	${Boost_INCLUDE_DIR} )

include_directories( ${INCLUDE_DIRS} )

add_subdirectory( phx )
add_subdirectory( mem )
add_subdirectory( util )
add_subdirectory( gfx )

add_executable( tests test_main )

target_link_libraries( tests phx_test util_test memmisc_test gfx_test
	${CUDA_LIBRARIES}
	${OPENGL_LIBRARY}
	${SDL_LIBRARY}
	${SDLIMAGE_LIBRARY}
	${CEGUI_LIBRARIES}
	${CEGUIOPENGL_LIBRARIES}
	${Boost_LIBRARIES} 
	${CPPUNIT_LIBRARIES} )

