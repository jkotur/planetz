#ifndef GPU_PLANET_H
#define GPU_PLANET_H

namespace GPU
{
	class Planet
	{
		public:
			Planet(uint32_t id);
			virtual ~Planet();

			float3 getPos() const;
			float getRadius() const;
			void setPos(float3 new_pos);
			void setRadius(float new_radius);
	}
}

#endif // GPU_PLANET_H
