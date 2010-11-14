cmake_minimum_required(VERSION 2.6)

project( inzynierka )

SET( CMAKE_MODULE_PATH
	${CMAKE_MODULE_PATH}
	${CMAKE_CURRENT_SOURCE_DIR}/cmake )

SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )

find_package( CUDA )

find_package( SDL )
find_package( SDL_image )

find_package( OpenGL )

find_package( Boost COMPONENTS filesystem regex signals )
find_package( CEGUI )
find_package( CEGUIOPENGL )

find_package( Sqlite3 )

if( UNIX )
	#	set( CMAKE_C_FLAGS "-O3 -g -Wall -pedantic")
	#	set( CMAKE_CXX_FLAGS "-O3 -g -Wall -pedantic")
	set( CMAKE_C_FLAGS "-O3 -g -Wall")
	set( CMAKE_CXX_FLAGS "-O3 -g -Wall")
endif( UNIX )

# is this needed?
set( CUDA_NVCC_FLAGS "-arch=sm_11" )

add_subdirectory( src )
add_subdirectory( unittest )

