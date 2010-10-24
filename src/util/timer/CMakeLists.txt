add_library( timer timer )

set( USING_BOOST true )

if( USING_BOOST )
	message(STATUS "Timer will use boost extensions")

	target_link_libraries( timer ${Boost_LIBRARY} )
else( USING_BOOST )
	message(STATUS "Timer will NOT use boost extensions")

	SET( CMAKE_CXX_FLAGS -D_NON_BOOST )
	target_link_libraries( timer )
endif( USING_BOOST )

