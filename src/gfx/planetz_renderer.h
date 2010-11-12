#ifndef __PLANETZ_RENDERER_H__

#define __PLANETZ_RENDERER_H__

#include "drawable.h"

#include "gpu/gfx_planet_factory.h"
#include "gpu/planet_model.h"

#include "shader.h"

namespace GFX {

class PlanetzRenderer : public Drawable {
public:
	PlanetzRenderer( const GPU::GfxPlanetFactory * factory );
	virtual ~PlanetzRenderer();
	
	virtual void draw() const;
	virtual void prepare();

	void setModels( GPU::PlanetzModel modPlanet );
private:
	Program pr;

	GPU::PlanetzModel modPlanet;

	GLint texModelId;

	const GPU::GfxPlanetFactory * const factory;
};

}

#endif /* __PLANETZ_RENDERER_H__ */

