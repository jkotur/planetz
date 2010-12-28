#ifndef _PHX_KERNELS_H_
#define _PHX_KERNELS_H_
#include <string>

//#define PHX_DEBUG
#ifdef PHX_DEBUG
__global__ void basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt, float3 *tmp_pos, float3 *tmp_vel, float* shrd, float *glbl, unsigned *where );
#else
__global__ void basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt, float3 *tmp_pos, float3 *tmp_vel );
#endif

std::string getErr();

#endif // _PHX_KERNELS_H_
