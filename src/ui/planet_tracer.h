#ifndef _DEBUG_PLANET_TRACER_H_
#define _DEBUG_PLANET_TRACER_H_
#include "util/vector.h"

namespace MEM
{
namespace MISC
{
	class PhxPlanetFactory;
}
}

namespace GFX
{
	class PlanetzPicker;
}

class Camera;

class PlanetTracer
{
	public:
		PlanetTracer( MEM::MISC::PhxPlanetFactory *f, GFX::PlanetzPicker *pp, Camera *cam );
		~PlanetTracer();
		
		bool on_button_down( int b, int x, int y );

		void refresh();

	private:

		MEM::MISC::PhxPlanetFactory *factory;
		GFX::PlanetzPicker *picker;
		Camera *camera;
		int id;
		Vector3 dist;
};

#endif // _DEBUG_PLANET_TRACER_H_

