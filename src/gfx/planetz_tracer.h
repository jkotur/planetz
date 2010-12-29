#ifndef __PLANET_TRACER_H__

#define __PLANET_TRACER_H__

#include "drawable.h"

#include "mem/misc/buffer.h"
#include "mem/misc/gfx_planet_factory.h"

namespace GFX
{
class PlanetsTracer : public Drawable {
	public:
		PlanetsTracer ( const MEM::MISC::GfxPlanetFactory& gpf
				, unsigned num = 10 , double freq = 1.0 )
			: gpf(gpf) , number(num) , oldest(0) , begin(0)
		{
		}

		virtual ~PlanetsTracer()
		{
		}

		void setLenght( unsigned int num )
		{
			number = num;
		}

		void update();
		virtual void update_configuration();
		virtual void draw() const;
		void clear();
	private:
		const MEM::MISC::GfxPlanetFactory& gpf;

		MEM::MISC::BufferGl<float3> positions;
		unsigned number;
		unsigned oldest;
		unsigned begin;
};

} // GFX

#endif /* __PLANET_TRACER_H__ */

