#ifndef __CUDA_UTIL_H__

#define __CUDA_UTIL_H__

#include <stdlib.h>
#include "util/logger.h"

#define CUT_CHECK_ERROR(errorMessage) do {                                \
	cudaThreadSynchronize();                                          \
	cudaError_t err = cudaGetLastError();                             \
	if( cudaSuccess != err) {                                         \
		log_printf(CRITICAL,                                      \
			"Cuda error: %s in file '%s' in line %i : %s.\n", \
			errorMessage, __FILE__, __LINE__,                 \
			cudaGetErrorString( err ) );                      \
		exit(EXIT_FAILURE);                                       \
	} } while (0)

#endif /* __CUDA_UTIL_H__ */

