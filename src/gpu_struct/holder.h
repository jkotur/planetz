#ifndef HOLDER_H
#define HOLDER_H

#include "buffer.h"

namespace GPU
{
	struct Holder
	{
		// Planet
		//   * GFX
		BufferGl<uint8_t>  planet_model;

		//   * COMMON
		BufferGl<float3>   planet_pos;
		BufferGl<float>    planet_radius;
		BufferGl<uint32_t> planet_count;

		//   * PHX
		BufferCu<float>    planet_mass;
		BufferCu<float3>   planet_velocity;

		BufferGl<float3>   pointsCloud_points;
		BufferGl<uint32_t> pointsCloud_size;
	};
}

#endif // HOLDER_H
