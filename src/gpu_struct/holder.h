#ifndef HOLDER_H
#define HOLDER_H

#include "buffer.h"

namespace GPU
{
	struct Holder
	{
		Buffer<float3> Planet_pos;
		Buffer<float> Planet_radius;
		Buffer<uint32_t> Planet_count;

		Buffer<float3> PointsCloud_points;
		Buffer<uint32_t> PointsCloud_size;
	};
}

#endif // HOLDER_H
