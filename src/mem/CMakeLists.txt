include_directories( . )

add_subdirectory( db )
add_subdirectory( misc )

add_library( mem memory_manager saver ioctl data_flow_mgr )
target_link_libraries( mem memmisc db ${CUDA_LIBRARIES} )

