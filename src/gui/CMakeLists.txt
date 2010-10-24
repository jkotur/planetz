include_directories( . )

add_library( gui gui layout planetzlayout )

target_link_libraries( gui ${CEGUI_LIBRARIES} )
