#ifndef __PLANETZ_RENDERER_H__

#define __PLANETZ_RENDERER_H__

#include "drawable.h"

#include "mem/misc/gfx_planet_factory.h"
#include "mem/misc/planet_model.h"

#include "geometry_renderer.h"
#include "copy_renderer.h"
#include "deffered_renderer.h"

namespace GFX {

class PlanetzRenderer : public Drawable {
public:
	PlanetzRenderer( const MEM::MISC::GfxPlanetFactory * factory );
	virtual ~PlanetzRenderer();
	
	virtual void draw() const;

	virtual void prepare();

	void setModels( MEM::MISC::PlanetzModel modPlanet );
private:
	GeomRender grend;
	CopyRender crend;
	DeferRender drend;

	const MEM::MISC::GfxPlanetFactory * const factory;
};

}

#endif /* __PLANETZ_RENDERER_H__ */

