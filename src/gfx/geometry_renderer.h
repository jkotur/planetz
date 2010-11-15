#ifndef __GEOMETRY_RENDERER_H__

#define __GEOMETRY_RENDERER_H__

#include "gpu/gfx_planet_factory.h"
#include "gpu/planet_model.h"
#include "shader.h"

namespace GFX
{

class GeomRender {
public:
	GeomRender( const GPU::GfxPlanetFactory * factory );
	virtual ~GeomRender();

	void setModels( GPU::PlanetzModel mod );

	void draw() const;

private:
	Program pr;

	GPU::PlanetzModel modPlanet;

	GLint texModelId;

	const GPU::GfxPlanetFactory * const factory;
};

} // GFX


#endif /* __GEOMETRY_RENDERER_H__ */

