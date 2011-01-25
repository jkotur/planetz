cmake_minimum_required(VERSION 2.6)

project( inzynierka )

cmake_policy(SET CMP0011 NEW)

SET( CMAKE_MODULE_PATH
	${CMAKE_MODULE_PATH}
	${CMAKE_CURRENT_SOURCE_DIR}/cmake )

SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
SET( PROJECT_INCLUDE_DIR
	${PROJECT_INCLUDE_DIR} 
	${PROJECT_SOURCE_DIR}/src/3rdparty/cudpp/include )

find_package( CUDA )

find_package( SDL )
find_package( SDL_image )

find_package( OpenGL )

find_package( Boost COMPONENTS filesystem regex signals program_options )
find_package( CEGUI )
find_package( CEGUIOPENGL )

find_package( Sqlite3 )


if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release"
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if( UNIX )
	set( CMAKE_C_FLAGS   "-O2")
	set( CMAKE_CXX_FLAGS "-O2")
	set( CMAKE_C_FLAGS_DEBUG   "-g -rdynamic -Wall -D_DEBUG" )
	set( CMAKE_CXX_FLAGS_DEBUG "-g -rdynamic -Wall -D_DEBUG" )
	set( CMAKE_C_FLAGS_RELEASE   "-D_RELEASE" )
	set( CMAKE_CXX_FLAGS_RELEASE "-D_RELEASE" )
endif( UNIX )

# is this needed?
set( CUDA_NVCC_FLAGS "-arch=sm_11" )

add_subdirectory( src )
add_subdirectory( unittest )

