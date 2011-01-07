#include "holder_cleaner_kernels.h"

__global__ void MEM::MISC::create_filter( float *masses, unsigned *filter, unsigned count )
{
	unsigned index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index >= count )
	{
		return;
	}
	filter[ index ] = masses[ index ] ? 1 : 0;
}
