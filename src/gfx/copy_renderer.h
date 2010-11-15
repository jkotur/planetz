#ifndef __COPY_RENDERER_H__

#define __COPY_RENDERER_H__

#include "mem/misc/gfx_planet_factory.h"

namespace GFX
{

class CopyRender {
public:
	CopyRender( const MEM::MISC::GfxPlanetFactory * factory );
	virtual ~CopyRender();

	void draw() const;
private:
		
	GLuint sphereListId;

	const MEM::MISC::GfxPlanetFactory * const factory;
};

} // GFX


#endif /* __COPY_RENDERER_H__ */

