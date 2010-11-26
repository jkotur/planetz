#ifndef _PHX_KERNELS_H_
#define _PHX_KERNELS_H_

__global__ void basic_interaction( float3 *positions, float* masses, float3 *velocities, unsigned *cnt );

#endif // _PHX_KERNELS_H_
