add_subdirectory( gui )
add_subdirectory( input )

include_directories( ${PROJECT_SOURCE_DIR}/src )

add_library( ui camera camera_manager ui planetz_picker )

target_link_libraries( ui gui input ${CEGUI_LIBRARIES} )
