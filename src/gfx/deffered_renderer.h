#ifndef __DEFFERED_RENDERER_H__

#define __DEFFERED_RENDERER_H__

#include "shader.h"
#include "drawable.h"
#include "mem/misc/buffer.h"
#include "mem/misc/gfx_planet_factory.h"

namespace GFX
{
class DeferRender : public Drawable {
public:
	DeferRender( const MEM::MISC::GfxPlanetFactory * factory );
	virtual ~DeferRender();

	virtual void draw() const;

	virtual void prepare();
	
private:
	GLuint generate_sphere_texture( int w , int h );
	GLuint generate_render_target_texture( int w , int h );

	Program prPlanet , prLighting;

	GLint screenId;
	GLint colourId;
	GLint radiusId;
	GLint sphereTexId;

	GLuint fboId;

	GLuint depthTex;
	GLuint colourTex;
	GLuint sphereTex;

	MEM::MISC::BufferGl<float> rect;
	MEM::MISC::BufferGl<float> rect2;
	
	const MEM::MISC::GfxPlanetFactory * const factory;
};

} // GFX


#endif /* __DEFFERED_RENDERER_H__ */

