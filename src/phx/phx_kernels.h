#ifndef _PHX_KERNELS_H_
#define _PHX_KERNELS_H_
#include <string>

//#define PHX_DEBUG
#ifdef PHX_DEBUG
__global__ void basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt, float3 *tmp_pos, float3 *tmp_vel, float3 *dvs, unsigned who );
__global__ void inside_cluster_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *shuffle, unsigned *counts, unsigned cluster, float3 *tmp_pos, float3 *tmp_vel, float3 *dvs, unsigned who, unsigned *whois );
#else
__global__ void basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt, float3 *tmp_pos, float3 *tmp_vel );
__global__ void inside_cluster_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *shuffle, unsigned *counts, unsigned cluster, float3 *tmp_pos, float3 *tmp_vel );
#endif


std::string getErr();

#endif // _PHX_KERNELS_H_
