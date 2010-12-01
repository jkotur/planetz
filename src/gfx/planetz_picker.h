#ifndef __PLANETZ_PICKER_H__

#define __PLANETZ_PICKER_H__

#include "mem/misc/gfx_planet_factory.h"
#include "mem/misc/buffer.h"
#include "gfx/shader.h"

namespace GFX
{
class PlanetzPicker {
public:
	PlanetzPicker( const MEM::MISC::GfxPlanetFactory * factory , int w , int h , int winw , int winh );
	virtual ~PlanetzPicker();

	void render( int x , int y );

	int getId();

	void resize( int w , int h );
	
private:
	void resizeNames();
	void generate_fb_textures();
	GLuint generate_sphere_texture( int w , int h );

	const MEM::MISC::GfxPlanetFactory * const factory;

	int w , h;
	int winw , winh;

	Shader vs , gs , fs;
	Program pr;

	GLuint fboId;

	GLuint depthTex;
	GLuint colorTex;

	GLuint sphereTex;

	GLint sphereTexId;

	float* buffNames;
	float* buffDepth;

	GLint radiusId , namesId;

	MEM::MISC::BufferGl<float> names;

	int max;
};

}

#endif /* __PLANETZ_PICKER_H__ */

