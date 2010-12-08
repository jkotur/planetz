include_directories( . )
include_directories( ./inc )

add_library( cutil src/bank_checker src/cmd_arg_reader src/cutil src/findFile src/multithreading src/param src/paramgl src/stopwatch src/stopwatch_linux )
target_link_libraries( cutil )
