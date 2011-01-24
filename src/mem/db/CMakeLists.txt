include_directories( . )

add_library( db db dbsqlite table row planet_row camera_row materials_row
	textures_row table_interface )
target_link_libraries( db )
