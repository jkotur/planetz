#ifndef HOLDER_H
#define HOLDER_H

#include "buffer.h"
#include "buffer_cu.hpp"

namespace GPU
{
	struct PlanetHolder
	{
		PlanetHolder();
		virtual ~PlanetHolder();

		void init( unsigned num = 0 );
		void resize(const size_t num);

		//   * GFX
		BufferGl<uint8_t>  model;

		//   * COMMON
		BufferGl<float3>   pos;
		BufferGl<float>    radius;
		BufferGl<uint32_t> count;

		//   * PHX
		BufferCu<float>    mass;
		BufferCu<float3>   velocity;
	};

	struct PointsCloudHolder
	{
		BufferCu<float>    phx_dt;

		BufferGl<float3>   pointsCloud_points;
		BufferGl<uint32_t> pointsCloud_size;
	};
}

#endif // HOLDER_H
