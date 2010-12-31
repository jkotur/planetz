#ifndef __PLANET_TRACER_H__

#define __PLANET_TRACER_H__

#include "drawable.h"

#include "util/config.h"
#include "util/timer/timer.h"

#include "mem/misc/buffer.h"
#include "mem/misc/gfx_planet_factory.h"

namespace GFX
{
class PlanetsTracer : public Drawable {
	public:
		PlanetsTracer( const MEM::MISC::GfxPlanetFactory& gpf
				, unsigned num = 10 , double freq = 1.0 )
			: gpf(gpf) , number(num) , oldest(0) , begin(0) , dt(freq)
		{
		}

		PlanetsTracer( const MEM::MISC::GfxPlanetFactory& gpf
		             , const Config& cfg )
			: gpf(gpf) , oldest(0) , begin(0)
		{
			update_configuration( cfg );
		}

		virtual ~PlanetsTracer()
		{
		}

		void setLenght( unsigned int num )
		{
			number = num;
		}

		void start();
		void stop();
		void clear();

		void update();
		virtual void update_configuration();
		void update_configuration( const Config& cfg );
		virtual void draw() const;
	private:
		const MEM::MISC::GfxPlanetFactory& gpf;

		Timer::Caller tc;

		MEM::MISC::BufferGl<float3> positions;
		unsigned number;
		unsigned oldest;
		unsigned begin;

		double dt;

		bool drawable;
};

} // GFX

#endif /* __PLANET_TRACER_H__ */

