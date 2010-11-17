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

	virtual void resize(
			unsigned int width ,
			unsigned int height );
	
private:
	void create_textures( unsigned int w , unsigned int h );
	void delete_textures();

	GLuint generate_sphere_texture( int w , int h );
	GLuint generate_render_target_texture( int w , int h );

	Program prPlanet , prLighting;

	static const GLsizei gbuffNum = 2;
	GLuint gbuffTex[gbuffNum];

	GLint gbuffId[gbuffNum];

	GLenum bufferlist[gbuffNum];

	GLint ratioId;

	GLint radiusId;
	GLint modelId ;
	GLint sphereTexId;

	GLuint fboId;

	GLuint depthTex;
	GLuint sphereTex;

	const MEM::MISC::GfxPlanetFactory * const factory;
};

} // GFX


#endif /* __DEFFERED_RENDERER_H__ */

