#ifndef __DEFFERED_RENDERER_H__

#define __DEFFERED_RENDERER_H__

#include "gfx.h"
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
	
	void setMaterials( GLuint );

	void on_camera_angle_changed( float*m );
private:
	void create_textures( unsigned int w , unsigned int h );
	void delete_textures();

	//
	// Shaders
	//
	Program prPlanet , prLighting , prLightsBase;

	//
	// Vertex data
	//
	GLint radiusId;

	GLint modelId ;
	GLint modelLId ;
	GLint emissiveLId;

	//
	// Sphere normals (deprecated)
	//
	GLuint generate_sphere_texture( int w , int h );

	GLint sphereTexId;
	GLuint sphereTex;
	
	//
	// Materials
	//
	GLint materialsTexId;
	GLuint materialsTex;
	GLuint matLId;

	//
	// MTR
	//
	GLuint generate_render_target_texture( int w , int h );

	GLuint fboId;
	GLuint depthTex;

	static const GLsizei gbuffNum = 4;

	GLint  gbuffId   [gbuffNum*2];
	GLuint gbuffTex  [gbuffNum  ];
	GLenum bufferlist[gbuffNum  ];

	//
	// Texturing
	// 
	GLuint generate_angles_texture( int w , int h );
	GLuint generate_normals_texture( int w , int h );

	GLuint anglesTex;
	GLuint normalsTex;

	Texture*texture;

	GLint anglesTexId;
	GLint normalsTexId;
	GLint textureTexId;

	GLint anglesId;

	const MEM::MISC::GfxPlanetFactory * const factory;
};

} // GFX


#endif /* __DEFFERED_RENDERER_H__ */

