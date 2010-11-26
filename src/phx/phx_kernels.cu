#include "phx_kernels.h"

__device__ const float dt = 1e-3f;

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

__device__ inline float3& operator+=( float3& l, const float3& r )
{
	l.x += r.x;
	l.y += r.y;
	l.z += r.z;
	return l;
}

__device__ float3 get_dV( float3 myPos, float3 theirPos, float theirMass )
{
	float3 dir = theirPos - myPos;
	float r2 = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
	return dir * (dt / ( r2 * sqrtf( r2 ) + 1e-3f) );
}

__global__ void basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt )
{
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned count = *cnt;
	if( index >= count )
	{
		return;
	}
	float3 new_pos = positions[ index ];
	float3 new_vel = velocities[ index ];

	for( unsigned i = 0; i < count; ++i )
	{
		if( i != index )
		{
			new_vel += get_dV( positions[ index ], positions[ i ], masses[ i ] );
			new_pos += new_vel * dt;
		}
	}

	__syncthreads();

	positions[ index ] = new_pos;
	velocities[ index ] = new_vel;
}
