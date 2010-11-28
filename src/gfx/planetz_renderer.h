#ifndef __PLANETZ_RENDERER_H__

#define __PLANETZ_RENDERER_H__

#include "drawable.h"

#include "mem/misc/gfx_planet_factory.h"
#include "mem/misc/planet_model.h"

#include "deffered_renderer.h"

namespace GFX {

class PlanetzRenderer : public Drawable {
public:
	PlanetzRenderer( const MEM::MISC::GfxPlanetFactory * factory );
	virtual ~PlanetzRenderer();
	
	virtual void draw() const;

	virtual void prepare();

	virtual void resize(
			unsigned int width ,
			unsigned int height );

	void setGfx( Gfx * _g );

	void setMaterials( GLuint matTex );
private:
	DeferRender drend;

	const MEM::MISC::GfxPlanetFactory * const factory;
};

}

#endif /* __PLANETZ_RENDERER_H__ */

