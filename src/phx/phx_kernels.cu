#include "phx_kernels.h"

__device__ const float dt = 2e-2f;
__device__ const float G = 10.0f;

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

__global__ void basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt )
{
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned count = *cnt;
	extern __shared__ float3 s_positions[];
	float *s_masses = (float*) (s_positions + blockDim.x);

	if( index >= count )
	{
		return;
	}
	float3 old_pos = positions[ index ];
	float3 new_vel = velocities[ index ];

	for( unsigned i = 0; i < count; i += blockDim.x )
	{
		if( threadIdx.x + i < count )
		{
			s_positions[ threadIdx.x ] = positions[ i + threadIdx.x ];
			s_masses[ threadIdx.x ] = masses[ i + threadIdx.x ];
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

	positions[ index ] = old_pos + new_vel * dt;
	velocities[ index ] = new_vel;
}

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

