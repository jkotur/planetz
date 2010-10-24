include_directories(
	${PROJECT_SOURCE_DIR}/src
	${PROJECT_SOURCE_DIR}/src/ui )

add_library( gui gui layout planetzlayout )

target_link_libraries( gui ${CEGUI_LIBRARIES} )
