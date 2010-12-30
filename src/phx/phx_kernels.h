#ifndef _PHX_KERNELS_H_
#define _PHX_KERNELS_H_
#include <string>

//#define PHX_DEBUG
#ifdef PHX_DEBUG
__global__ void basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt, float3 *tmp_pos, float3 *tmp_vel, float3 *dvs, unsigned who );
__global__ void inside_cluster_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *shuffle, unsigned *counts, unsigned cluster, float3 *dvs, unsigned who, unsigned *whois );
#else
__global__ void basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt, float3 *tmp_pos, float3 *tmp_vel );
__global__ void inside_cluster_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *shuffle, unsigned *counts, unsigned cluster );
#endif
/// @brief Oblicza prędkości całych klastrów
__global__ void outside_cluster_interaction( float3 *centers, float *masses, unsigned count, float3 *velocities_impact );

/// @brief Propaguje dV klastrów do planet w tych klastrach
__global__ void propagate_velocities( float3 *velocities_impact, float3 *positions, float3 *velocities, unsigned *shuffle, unsigned *count, unsigned last_cluster );

/// @brief Oblicza nowe pozycje na podstawie prędkości
__global__ void update_positions_kernel( float3 *positions, float3 *velocities, unsigned *count );

std::string getErr();

#endif // _PHX_KERNELS_H_
