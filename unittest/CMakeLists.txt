cmake_policy(VERSION 2.6)

find_package( cppunit )

set( INCLUDE_DIRS
	.
	../src
	${CUDA_INCLUDE_DIRS}
	${OPENGL_INCLUDE_DIR}
	${SDL_INCLUDE_DIR}
	${CEGUI_INCLUDE_DIR}
	${Boost_INCLUDE_DIR} )

include_directories( ${INCLUDE_DIRS} )

add_subdirectory( phx )

add_executable( tests test_main )

target_link_libraries( tests phx_test
	${CUDA_LIBRARIES}
	${OPENGL_LIBRARY}
	${SDL_LIBRARY}
	${SDLIMAGE_LIBRARY}
	${CEGUI_LIBRARIES}
	${CEGUIOPENGL_LIBRARIES}
	${Boost_LIBRARIES} 
	${cppunit_LIBRARIES} )

