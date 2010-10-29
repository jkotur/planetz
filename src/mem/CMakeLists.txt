include_directories( . )

add_library( mem memory_manager saver )
target_link_libraries( mem gpu ${CUDA_LIBRARIES} )

