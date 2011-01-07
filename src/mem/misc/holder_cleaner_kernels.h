#ifndef __HOLDER_CLEANER_KERNELS_H__
#define __HOLDER_CLEANER_KERNELS_H__

#include "cuda/reduce.cu"

namespace MEM
{
namespace MISC
{
	/**
	 * @brief Tworzy filtr do usuwania planet.
	 * 
	 * @details Usunięte zostaną planety, których masa wynosi 0.
	 */
	__global__ void create_filter( float *masses, unsigned *filter, unsigned count );
}
}

#endif // __HOLDER_CLEANER_KERNELS_H__
