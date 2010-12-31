#include "holder_kernels.h"

__global__ void stretch_kernel( unsigned *in_data, unsigned *out_data, unsigned in_data_size, unsigned factor )
{
	unsigned index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index >= in_data_size )
	{
		return;
	}
	unsigned r_factor = factor; // potrzebne? czy argumenty kernela już są w rejestrach?
	for( unsigned i = 0; i < r_factor; ++i )
	{
		out_data[ r_factor * index + i ] = in_data[ index ];
	}
}

void stretch( MEM::MISC::BufferCu<unsigned> *in, MEM::MISC::BufferCu<unsigned> *out, unsigned factor )
{
	unsigned threads = in->getLen();
	out->resize( threads * factor );
	dim3 block( min( 512, threads ) );
	dim3 grid( 1 + ( threads - 1 ) / block.x );
	stretch_kernel<<<grid, block>>>(
		in->d_data(),
		out->d_data(),
		threads,
		factor );
	CUT_CHECK_ERROR("kernel launch");
}
