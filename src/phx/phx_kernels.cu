#include "phx_kernels.h"
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

__device__ inline float3 operator+( const float3 &l , const float3 &r )
{
	return make_float3( l.x + r.x, l.y + r.y, l.z + r.z );
}
__device__ inline float3 operator-( const float3 &l , const float3 &r )
{
	return make_float3( l.x - r.x, l.y - r.y, l.z - r.z );
}

__device__ inline float3 operator/( const float3 &v , const float &f )
{
	return make_float3( v.x / f, v.y / f, v.z / f );
}

__device__ inline float3 operator*( const float3 &v , const float &f )
{
	return make_float3( v.x * f, v.y * f, v.z * f );
}

__device__ inline /*float3&*/void operator+=( float3& l, const float3& r )
{
	l.x += r.x;
	l.y += r.y;
	l.z += r.z;
//	return l;
}

__device__ float3 get_dV( float3 myPos, float3 theirPos, float theirMass )
{
	float3 dir = theirPos - myPos;
	float r2 = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
	if( r2 < 1 ) r2 = 1; //return dir / sqrtf( r2 ) * dt;
	return dir / r2 * G * (dt / sqrtf( r2 ) );
}

#ifndef PHX_DEBUG
__global__ void basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt, float3 *tmp_pos, float3 *tmp_vel )
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
				// don't interact with yourself
				if( other_index != index )
				{
					//new_vel += get_dV( old_pos, s_positions[j], s_masses[j] );
					new_vel += get_dV( old_pos, positions[other_index], masses[other_index] );
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

#else
__global__ void basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt, float3 *tmp_pos, float3 *tmp_vel, float* shrd, float *glbl, unsigned *where )
{
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
	if( !index ) error[0] = '\0';
	unsigned count = *cnt;
	/*extern __shared__ float3 s_positions[];
	float *s_masses = (float*) (s_positions + blockDim.x);
	*/
	__shared__ float3 s_positions[512];
	__shared__ float s_masses[512];

	float3 old_pos, new_vel;

	if( index < count )
	{
		old_pos = positions[ index ];
		new_vel = velocities[ index ];
	}

	for( unsigned i = 0; i < count; i += blockDim.x )
	{
		if( threadIdx.x + i < count )
		{
			s_positions[ threadIdx.x ] = positions[ i + threadIdx.x ];
			s_masses[ threadIdx.x ] = masses[ i + threadIdx.x ];
			if( blockIdx.x == gridDim.x - 1 )
			{
				CUDA_ASSERT( where[ i + threadIdx.x ] == 0 );
				where[ i + threadIdx.x ] = 1337;
			}
			CUDA_ASSERT( fabs( s_masses[threadIdx.x] - masses[i + threadIdx.x] ) < 1e-3 );
			CUDA_ASSERT( fabs( s_positions[threadIdx.x].x - positions[i + threadIdx.x].x ) < 1e-3 );
		}
		__syncthreads();
		if( index < count )  // <- nadmiarowe wątki tylko do kopiowania danych
		{
			for( unsigned j = 0; j < blockDim.x && i + j < count; ++j )
			{
				if( i + j != index )
				{
					CUDA_ASSERT( where[ i + j ] );
					if( blockIdx.x == gridDim.x - 1 && threadIdx.x == 0 )
					{
						//where[i+j] = 666;
					}
					if( fabs( s_masses[j] - masses[i + j] ) >= 1e-3 )
					{
					if( index == 1 )
					{
						*shrd = s_masses[j];
						*glbl = masses[i + j];
						//where[i+j] = 123;
					}
					}
			//		CUDA_ASSERT( fabs( s_masses[j] - masses[i + j] ) < 1e-3 );
				//	CUDA_ASSERT( fabs( s_positions[j].x - positions[i + j].x ) < 1e-3 );
					new_vel += get_dV( old_pos, s_positions[ j ], s_masses[ j ] );
				}
			}
		}
		__syncthreads();
	}
	if( index >= count )
	{
		return;
	}

	tmp_pos[ index ] = old_pos + new_vel * dt;
	tmp_vel[ index ] = new_vel;
}
#endif

#if 0 // Poprawić, kiedy basic_interaction będzie już działać
__global__ void inside_cluster_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *shuffle, unsigned *counts, unsigned cluster )
{
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned mapped_index = shuffle[ index ];
	unsigned count = counts[ cluster ];
	extern __shared__ float3 s_positions[];
	float *s_masses = (float*) (s_positions + blockDim.x);

	if( index >= count )
	{
		return;
	}
	float3 old_pos = positions[ mapped_index ];
	float3 new_vel = velocities[ mapped_index ];

	for( unsigned i = (cluster ? counts[cluster-1] : 0); i < count; i += blockDim.x )
	{
		if( threadIdx.x + i < count )
		{
			s_positions[ threadIdx.x ] = positions[ shuffle[ i + threadIdx.x ] ];
			s_masses[ threadIdx.x ] = masses[ shuffle[ i + threadIdx.x ] ];
		}
		__syncthreads();
		for( unsigned j = 0; j < blockDim.x && i + j < count; ++j )
		{
			if( i + j != index )
			{
				new_vel += get_dV( old_pos, s_positions[ j ], s_masses[ j ] );
			}
		}
		__syncthreads();
	}
	
	positions[ mapped_index ] = old_pos + new_vel * dt;
	velocities[ mapped_index ] = new_vel;
}
#endif
