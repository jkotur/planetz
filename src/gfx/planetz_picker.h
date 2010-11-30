#ifndef __PLANETZ_PICKER_H__

#define __PLANETZ_PICKER_H__

#include "mem/misc/gfx_planet_factory.h"
#include "mem/misc/buffer.h"
#include "gfx/shader.h"

namespace GFX
{
class PlanetzPicker {
public:
	PlanetzPicker( const MEM::MISC::GfxPlanetFactory * factory , int w , int h );
	virtual ~PlanetzPicker();

	void render( int x , int y );

	int getId();
	
private:
	void resizeNames();

	const MEM::MISC::GfxPlanetFactory * const factory;

	int w , h;

	Shader vs , gs , fs;
	Program pr;

	GLuint fboId;

	GLuint depthTex;
	GLuint colorTex;

	float* buffNames;
	float* buffDepth;

	GLint radiusId , namesId;

	MEM::MISC::BufferGl<float> names;

	int max;
};

}

#endif /* __PLANETZ_PICKER_H__ */

