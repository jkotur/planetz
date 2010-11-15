#ifndef __DEFFERED_RENDERER_H__

#define __DEFFERED_RENDERER_H__

#include "gpu/gfx_planet_factory.h"
#include "shader.h"

namespace GFX
{
class DeferRender {
public:
	DeferRender( const GPU::GfxPlanetFactory * factory );
	virtual ~DeferRender();

	void draw() const;
	
private:
	Program pr;

	GLint radiusId;
	
	const GPU::GfxPlanetFactory * const factory;
};

} // GFX


#endif /* __DEFFERED_RENDERER_H__ */

