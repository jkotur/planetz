include_directories( . include src ../cutil/inc )

CUDA_ADD_LIBRARY( cudpp
	src/cudpp
	src/cudpp_maximal_launch
	src/cudpp_plan
	src/cudpp_plan_manager
	src/app/compact_app.cu
	src/app/radixsort_app.cu
	src/app/rand_app.cu
	src/app/scan_app.cu
	src/app/segmented_scan_app.cu
	src/app/spmvmult_app.cu )
target_link_libraries( cudpp )
