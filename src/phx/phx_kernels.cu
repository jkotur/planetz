#include "phx_kernels.h"
#include "cuda/math.h"

#define ERROR_LEN 256
#define CUDA_ASSERT( x ) \
if( !(x) ) { if( error[0] != '\0' ) return; unsigned i = 0; while( #x[i] != '\0' ){error[i] = #x[i]; ++i;} error[i] = '\0'; return; }

__device__ const float dt = 2e-2f;
__device__ const float G = 1.f;
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
	return theirMass * dir / sqrtf( r2 ) * ( G * dt / r2 );
}

#ifndef PHX_DEBUG
__global__ void PHX::basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt )
#else
__global__ void PHX::basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt, float3 *dvs, unsigned who )
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

	velocities[ index ] += new_vel;
}

#ifndef PHX_DEBUG
__global__ void PHX::inside_cluster_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *shuffle, unsigned *counts, unsigned cluster )
#else
__global__ void PHX::inside_cluster_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *shuffle, unsigned *counts, unsigned cluster, float3 *dvs, unsigned who, unsigned *whois )
#endif
{
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned offset = cluster ? counts[ cluster-1 ] : 0;
	index += offset;
	unsigned mapped_index = shuffle[ index ];
	unsigned count = counts[ cluster ];

	float3 vel_diff;
	float3 old_pos;

	if( index >= count )
	{
		return;
	}
	old_pos = positions[ mapped_index ];
	vel_diff = make_float3(0,0,0);
	
	for( unsigned i = offset; i < count; i += blockDim.x )
	{
		for( unsigned j = 0; j < blockDim.x && i + j < count; ++j )
		{
			unsigned other_index = shuffle[ i + j ];
			if( other_index != mapped_index )
			{
#ifndef PHX_DEBUG
				vel_diff += get_dV( old_pos, positions[other_index], masses[other_index] );
#else
				float3 dv = get_dV( old_pos, positions[other_index], masses[other_index] );
				if( index == who )
				{
					dvs[ i + j ] = dv;
				}
				vel_diff += dv;
#endif
			}
		}
	}
	
	velocities[ mapped_index ] += vel_diff;
#ifdef PHX_DEBUG
	CUDA_ASSERT( whois[ mapped_index ] == 0 );
	whois[ mapped_index ] = index;
#endif
}

__global__ void PHX::outside_cluster_interaction( float3 *centers, float *masses, unsigned count, float3 *velocities_impact )
{
	unsigned tid = threadIdx.x;
	
	float3 pos = centers[tid];
	float3 new_vel = make_float3(0,0,0);

	for( unsigned i = 0; i < count; ++i )
	{
		if( i != tid )
		{
			new_vel += get_dV( pos, centers[i], masses[i] );
		}
	}
	velocities_impact[tid] = new_vel;
}

__global__ void PHX::propagate_velocities( float3 *velocities_impact, float3 *positions, float3 *velocities, unsigned *shuffle, unsigned *count, unsigned last_cluster )
{
	unsigned index = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned cluster = 0;

	if( index >= count[last_cluster] )
	{
		return;
	}

	// znajdujemy nasz klaster
	while( count[cluster] <= index ) ++cluster;

	// i zwiększamy swoją prędkość
	velocities[ shuffle[ index ] ] += velocities_impact[ cluster ];
}

__global__ void PHX::update_positions_kernel( float3 *positions, float3 *velocities, unsigned *count )
{
	unsigned index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index < *count )
	{
		positions[ index ] += velocities[ index ] * dt;
	}
}

__device__ bool collision_detected( float3 pos1, float r1, float3 pos2, float r2 )
{
	if( r1 == 0 || r2 == 0 ) return false;
	float3 dp = pos1 - pos2;
	float d2 = dp.x*dp.x + dp.y*dp.y + dp.z*dp.z; //kwadrat odległości środków
	return d2 < (r1+r2)*(r1+r2); // TODO coverage?
}

__global__ void PHX::detect_collisions( float3 *positions, float *radiuses, unsigned *count, unsigned *shuffle, unsigned last_cluster, unsigned *merges, unsigned *merge_needed )
{
	unsigned index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index >= count[last_cluster] )
	{
		return;
	}
	unsigned mapped_index = shuffle[ index ];
	float3 my_pos = positions[ mapped_index ];
	float my_radius = radiuses[ mapped_index ];
	unsigned cluster = 0;
	
	// znajdujemy nasz klaster
	while( count[cluster] <= index ) ++cluster;
	unsigned limit = count[cluster];
	
	for( unsigned i = index + 1; i < limit; ++i )
	{
		unsigned other_index = shuffle[i];
		if( collision_detected( my_pos, my_radius, positions[other_index], radiuses[other_index] ) )
		{
			merges[ mapped_index ] = other_index;
			*merge_needed = 1;
			return;
		}
	}
	merges[ mapped_index ] = mapped_index; // brak kolizji
}

__global__ void PHX::detect_collisions_no_clusters( float3 *positions, float *radiuses, unsigned count, unsigned *merges, unsigned *merge_needed )
{
	unsigned index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index >= count )
	{
		return;
	}
	float3 my_pos = positions[ index ];
	float my_radius = radiuses[ index ];
	
	for( unsigned i = index + 1; i < count; ++i )
	{
		if( collision_detected( my_pos, my_radius, positions[i], radiuses[i] ) )
		{
			merges[ index ] = i;
			*merge_needed = 1;
			return;
		}
	}
	merges[ index ] = index; // brak kolizji
}


__device__ void internal_merge( float3 *positions, float3 *velocities, float *masses, float *radiuses, unsigned idx1, unsigned idx2 )
{
	// wynik sklejenia ląduje w idx1, więc być może trzeba je zamienić
	if( radiuses[idx1] < radiuses[idx2] )
	{
		unsigned tmp = idx1;
		idx1 = idx2;
		idx2 = tmp;
	}

	float a1 = ( radiuses[idx1] * masses[idx1] ) / ( (radiuses[idx1] * masses[idx1]) + (radiuses[idx2] * masses[idx2]) );
	float b1 = masses[idx1] / ( masses[idx1] + masses[idx2] );
	positions[idx1] = positions[idx1] * a1 + positions[idx2] * (1-a1);
	velocities[idx1] = velocities[idx1] * b1 + velocities[idx2] * (1-b1);
	masses[idx1] += masses[idx2];
	radiuses[idx1] = powf( powf(radiuses[idx1], 3.) + powf(radiuses[idx2], 3.), 1.f/3.f );

	// oznacz jako skasowaną
	masses[idx2] = 0;
	radiuses[idx2] = 0;
}

__global__ void PHX::merge_collisions( unsigned *in_merges, unsigned *out_merges, float3 *positions, float3 *velocities, float *masses, float *radiuses, unsigned *count, unsigned *done )
{
	unsigned index = threadIdx.x + blockDim.x * blockIdx.x;
	if( index >= *count )
	{
		return;
	}
	unsigned to_merge = in_merges[ index ];
	
	if( index == to_merge )
	{
		out_merges[ index ] = in_merges[ index ];
		return;
	}

	if( in_merges[ to_merge ] != to_merge )
	{
		// ups, nasz kandydat na planetę znalazł kogoś innego - czekamy na lepsze czasy
		out_merges[ index ] = in_merges[ index ];
		*done = 0;
		return;
	}

	// jeżeli dotarliśmy tutaj, to już można mergować
	internal_merge( positions, velocities, masses, radiuses, index, to_merge );

	out_merges[ index ] = index;
}

