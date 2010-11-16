#ifndef __GEOMETRY_RENDERER_H__

#define __GEOMETRY_RENDERER_H__

#include "mem/misc/gfx_planet_factory.h"
#include "mem/misc/planet_model.h"
#include "shader.h"
#include "drawable.h"

namespace GFX
{

class GeomRender : public Drawable {
public:
	GeomRender( const MEM::MISC::GfxPlanetFactory * factory );
	virtual ~GeomRender();

	void setModels( MEM::MISC::PlanetzModel mod );

	void draw() const;

	void prepare();
private:
	Program pr;

	MEM::MISC::PlanetzModel modPlanet;

	GLint texModelId;

	const MEM::MISC::GfxPlanetFactory * const factory;
};

} // GFX


#endif /* __GEOMETRY_RENDERER_H__ */

