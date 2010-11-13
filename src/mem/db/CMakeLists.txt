include_directories( . )

add_library( db db dbsqlite table row planet_row table_interface )
target_link_libraries( db )
