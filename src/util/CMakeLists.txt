add_subdirectory( timer )

add_library( util logger vector locate )

target_link_libraries( util timer ${Boost_LIBRARY} )
