#ifndef __DEFFERED_RENDERER_H__

#define __DEFFERED_RENDERER_H__

#include "gfx.h"
#include "shader.h"
#include "drawable.h"
#include "mem/misc/buffer.h"
#include "mem/misc/gfx_planet_factory.h"

namespace GFX
{
/** 
 * @brief Klasa odpowiedzialna za wyświetlanie planet na ekran,
 * oraz wszystkich efektów towarzyszących.
 */
class DeferRender : public Drawable {
	enum OPT {
		LIGHTING    = 1 << 0,
		LIGHT_PLANES= 1 << 1,
		TEXTURES    = 1 << 2,
	};
public:
	/** 
	 * @brief Tworzy pustą, niegotową do pracy klasę.
	 * 
	 * @param factory
	 */
	DeferRender( const MEM::MISC::GfxPlanetFactory * factory );
	virtual ~DeferRender();

	virtual void draw() const;

	virtual void prepare();

	virtual void resize(
			unsigned int width ,
			unsigned int height );

	virtual void update_configuration();
	
	/** 
	 * @brief Ustawia teksturę przetrzymującą informację o materiałach.
	 * Musi to być jednowymiarowa tekstura.
	 * 
	 * @param id id tekstury
	 */
	void setMaterials( GLuint id );
	/** 
	 * @brief Ustawia teksturę tekstur planet. Wymagana jest dwuwymiarowa
	 * tablica tekstur (GL_TEXTURE_2D_ARRAY), która w kolejnych warstwach
	 * przetrzymuje tekstury planet.
	 * 
	 * @param id id tekstury
	 */
	void setTextures ( GLuint id );

	/** 
	 * @brief Funkcja wywoływana za każdym razem gdy zmieni się kąt patrzenia
	 * kamery. Konieczne do poprawnego teksturowania planet.
	 * 
	 * @param m macierz 4x4 obrotu kamery.
	 */
	void on_camera_angle_changed( float*m );
private:
	void create_textures( unsigned int w , unsigned int h );
	void delete_textures();

	//
	// Shaders
	//
	Program prPlanet , prLighting , prLightsBase , prAtmosphere , prPostAtm;

	//
	// Vertex data
	//
	GLint radiusId;

	GLint modelId ;
	GLint texIdId ;
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

	GLuint fboId[3];
	GLuint depthTex;

	static const GLsizei gbuffNum = 4;

	GLint  gbuffId   [gbuffNum*2+3];
	GLuint gbuffTex  [gbuffNum    ];
	GLenum bufferlist[gbuffNum    ];

	GLuint screenTex;

	//
	// Texturing
	// 
	GLuint generate_angles_texture( int w , int h );
	GLuint generate_normals_texture( int w , int h );

	GLuint anglesTex;
	GLuint normalsTex;
	GLuint texturesTex;

	GLint anglesTexId;
	GLint normalsTexId;
	GLint textureTexId;

	GLint anglesId;

	const MEM::MISC::GfxPlanetFactory * const factory;

	//
	// Atmospehere
	//
	GLuint generate_atmosphere_texture( int w , int h );

	GLint atmId;
	GLint atmDataId  , atmAtmDataId ;
	GLint atmColorId , atmAtmColorId;
	GLuint atmTex;

	GLint atmMaterialsId;
	GLint atmModelId;
	GLint atmEmissiveId;

	//
	// Glow
	//
	void generate_glow_planes( MEM::MISC::BufferGl<float>& buf , int num , int size );

//        MEM::MISC::BufferGl<float> planes;

	static const int glow_size = 128;

	Texture*tmptex;

	//
	// options switches
	//
	unsigned flags;

	GLint ifplanesId;
	GLint iftexturesId;
	GLint ifnormalsId;
	GLint brightness;
};

} // GFX


#endif /* __DEFFERED_RENDERER_H__ */

