include_directories( . )

add_subdirectory( db )

add_library( mem memory_manager saver )
target_link_libraries( mem gpu db ${CUDA_LIBRARIES} )

