#ifndef __COPY_RENDERER_H__

#define __COPY_RENDERER_H__

#include "gpu/gfx_planet_factory.h"

namespace GFX
{

class CopyRender {
public:
	CopyRender( const GPU::GfxPlanetFactory * factory );
	virtual ~CopyRender();

	void draw() const;
private:
		
	GLuint sphereListId;

	const GPU::GfxPlanetFactory * const factory;
};

} // GFX


#endif /* __COPY_RENDERER_H__ */

