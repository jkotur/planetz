include_directories( . )

add_library( cudpp src/cudpp src/cudpp_maximal_launch src/cudpp_plan src/cudpp_plan_manager )
target_link_libraries( cudpp )
