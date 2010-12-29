#include "phx_kernels.h"
#include "cuda/math.h"
#define ERROR_LEN 256
#define CUDA_ASSERT( x ) \
if( !(x) ) { if( error[0] != '\0' ) return; unsigned i = 0; while( #x[i] != '\0' ){error[i] = #x[i]; ++i;} error[i] = '\0'; return; }

__device__ const float dt = 2e-2f;
__device__ const float G = 1e3f;
__device__ char error[ERROR_LEN] = "";

__global__ void getErr( char *err )
{
	for( unsigned i = 0; i < ERROR_LEN; ++i )
	{
		err[i] = error[i];
	}
	error[0] = '\0';
}

std::string getErr()
{
	char *d_err;
	cudaMalloc( &d_err, sizeof(char) * ERROR_LEN );
	getErr<<<1, 1>>>( d_err );
	char h_err[256];
	cudaMemcpy( h_err, d_err, sizeof(char) * 256, cudaMemcpyDeviceToHost );
	cudaFree( d_err );
	return h_err;
}

__device__ float3 get_dV( float3 myPos, float3 theirPos, float theirMass )
{
	float3 dir = theirPos - myPos;
	float r2 = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
	if( r2 < 1 ) r2 = 1; //return dir / sqrtf( r2 ) * dt;
	return dir / sqrtf( r2 )  * ( G * dt / r2 );
}

#ifndef PHX_DEBUG
__global__ void basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt, float3 *tmp_pos, float3 *tmp_vel )
#else
__global__ void basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt, float3 *tmp_pos, float3 *tmp_vel, float3 *dvs, unsigned who )
#endif
{
	// shared mem temporarily turned off
	//__shared__ float3 s_positions[512];
	//__shared__ float s_masses[512];

	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned count = *cnt;

	float3 new_vel;
	float3 old_pos;

	if( index < count )
	{
		old_pos = positions[ index ];
		new_vel = make_float3(0,0,0);
	}
	
	for( unsigned i = 0; i < gridDim.x; ++i )
	{
		// copy blockDim.x data from global to shared mem
		/*
		if( i + threadIdx.x < count )
		{
			s_positions[ threadIdx.x ] = positions[ i + threadIdx.x ];
			s_masses[ threadIdx.x ] = masses[ i + threadIdx.x ];
		}
		__syncthreads();
		*/
		// use shared memory to calculate partial dV (interaction with planets [i..i+blockDim.x] )
		if( index < count )
		{
			for( unsigned j = 0; j < blockDim.x; ++j )
			{
				unsigned other_index = i * blockDim.x + j;
				if( other_index >= count ) break;
				// don't interact with yourself
				if( other_index != index )
				{
					//new_vel += get_dV( old_pos, s_positions[j], s_masses[j] );
#ifndef PHX_DEBUG
					new_vel += get_dV( old_pos, positions[other_index], masses[other_index] );
#else
					float3 dv = get_dV( old_pos, positions[other_index], masses[other_index] );
					if( index == who )
					{
						dvs[ other_index ] = dv;
					}
					new_vel += dv;
#endif
				}
			}
		}

	}
	__syncthreads();
	if( index >= count )
	{
		return;
	}

	new_vel += velocities[ index ];
	tmp_pos[ index ] = old_pos + new_vel * dt;
	tmp_vel[ index ] = new_vel;
}

#ifndef PHX_DEBUG
__global__ void inside_cluster_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *shuffle, unsigned *counts, unsigned cluster, float3 *tmp_pos, float3 *tmp_vel )
#else
__global__ void inside_cluster_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *shuffle, unsigned *counts, unsigned cluster, float3 *tmp_pos, float3 *tmp_vel, float3 *dvs, unsigned who, unsigned *whois )
#endif
{
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned offset = cluster ? counts[ cluster-1 ] : 0;
	index += offset;
	unsigned mapped_index = shuffle[ index ];
	unsigned count = counts[ cluster ];

	float3 new_vel;
	float3 old_pos;

	if( index >= count )
	{
		return;
	}
	old_pos = positions[ mapped_index ];
	new_vel = make_float3(0,0,0);
	
	for( unsigned i = offset; i < count; i += blockDim.x )
	{
		for( unsigned j = 0; j < blockDim.x && i + j < count; ++j )
		{
			unsigned other_index = shuffle[ i + j ];
			if( other_index != mapped_index )
			{
#ifndef PHX_DEBUG
				new_vel += get_dV( old_pos, positions[other_index], masses[other_index] );
#else
				float3 dv = get_dV( old_pos, positions[other_index], masses[other_index] );
				if( index == who )
				{
					dvs[ i + j ] = dv;
				}
				new_vel += dv;
#endif
			}
		}
	}
	
	new_vel += velocities[ mapped_index ];
	tmp_pos[ mapped_index ] = old_pos + new_vel * dt;
	tmp_vel[ mapped_index ] = new_vel;
#ifdef PHX_DEBUG
	CUDA_ASSERT( whois[ mapped_index ] == 0 );
	whois[ mapped_index ] = index;
#endif
}
