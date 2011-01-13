#include "deffered_renderer.h"

#include <GL/glew.h>
#include <cmath>
#include <algorithm>

#include "gfx.h"

#include "constants.h"

using namespace GFX;

DeferRender::DeferRender( const MEM::MISC::GfxPlanetFactory * factory )
	: materialsTex(0) , factory(factory) , flags(0)
{
}

DeferRender::~DeferRender()
{
	delete_textures();
}

void DeferRender::setMaterials( GLuint mat )
{
	materialsTex = mat;
}

void DeferRender::setTextures ( GLuint tex )
{
	texturesTex = tex;
}

void DeferRender::prepare()
{
	prPlanet.create(
		gfx->shmMgr.loadShader(GL_VERTEX_SHADER  ,
			DATA("shaders/deffered_01.vert")),
		gfx->shmMgr.loadShader(GL_FRAGMENT_SHADER,
			DATA("shaders/deffered_01.frag")),
		gfx->shmMgr.loadShader(GL_GEOMETRY_SHADER,
			DATA("shaders/deffered_01.geom")),
		GL_POINTS , GL_QUAD_STRIP );

	prLightsBase.create(
		gfx->shmMgr.loadShader(GL_VERTEX_SHADER  ,
			DATA("shaders/deffered_02.vert")),
		gfx->shmMgr.loadShader(GL_FRAGMENT_SHADER,
			DATA("shaders/deffered_02.frag")),
		gfx->shmMgr.loadShader(GL_GEOMETRY_SHADER,
			DATA("shaders/deffered_02.geom")),
		GL_POINTS , GL_QUAD_STRIP );

	prLighting.create(
		gfx->shmMgr.loadShader(GL_VERTEX_SHADER  ,
			DATA("shaders/deffered_03.vert")),
		gfx->shmMgr.loadShader(GL_FRAGMENT_SHADER,
			DATA("shaders/deffered_03.frag")),
		gfx->shmMgr.loadShader(GL_GEOMETRY_SHADER,
			DATA("shaders/deffered_03.geom")),
		GL_POINTS , GL_QUAD_STRIP );

	prAtmosphere.create(
		gfx->shmMgr.loadShader(GL_VERTEX_SHADER  ,
			DATA("shaders/deffered_04.vert")),
		gfx->shmMgr.loadShader(GL_FRAGMENT_SHADER,
			DATA("shaders/deffered_04.frag")),
		gfx->shmMgr.loadShader(GL_GEOMETRY_SHADER,
			DATA("shaders/deffered_04.geom")),
		GL_POINTS , GL_QUAD_STRIP );

	prPostAtm.create(
		gfx->shmMgr.loadShader(GL_VERTEX_SHADER  ,
			DATA("shaders/deffered_03.vert")),
		gfx->shmMgr.loadShader(GL_FRAGMENT_SHADER,
			DATA("shaders/deffered_05.frag")),
		gfx->shmMgr.loadShader(GL_GEOMETRY_SHADER,
			DATA("shaders/deffered_03.geom")),
		GL_POINTS , GL_QUAD_STRIP );

	sphereTexId    = glGetUniformLocation( prPlanet.id() , "sph_pos"     );
	materialsTexId = glGetUniformLocation( prPlanet.id() , "materialsTex");
	normalsTexId   = glGetUniformLocation( prPlanet.id() , "normalsTex"  );
	textureTexId   = glGetUniformLocation( prPlanet.id() , "texturesTex" );
	anglesId       = glGetUniformLocation( prPlanet.id() , "angles"      );
	radiusId       = glGetAttribLocation ( prPlanet.id() , "radius"      );
	modelId        = glGetAttribLocation ( prPlanet.id() , "model"       );
	texIdId        = glGetAttribLocation ( prPlanet.id() , "texId"       );
	atmDataId      = glGetAttribLocation ( prPlanet.id() , "atmData"     );
	atmColorId     = glGetAttribLocation ( prPlanet.id() , "atmColor"    );
                                                                             
	gbuffId[0]     = glGetUniformLocation( prLighting.id() , "gbuff1"    );
	gbuffId[1]     = glGetUniformLocation( prLighting.id() , "gbuff2"    );
	gbuffId[2]     = glGetUniformLocation( prLighting.id() , "gbuff3"    );
	gbuffId[3]     = glGetUniformLocation( prLighting.id() , "gbuff4"    );
	matLId         = glGetUniformLocation( prLighting.id() , "materials" );
	ifplanesId     = glGetUniformLocation( prLighting.id() , "planes"    );
	modelLId       = glGetAttribLocation ( prLighting.id() , "model"     );
	lightId        = glGetAttribLocation ( prLighting.id() , "light"     );
                                                                             
	gbuffId[4]     = glGetUniformLocation( prLightsBase.id() , "gbuff1"  );
	gbuffId[5]     = glGetUniformLocation( prLightsBase.id() , "gbuff2"  );
	gbuffId[6]     = glGetUniformLocation( prLightsBase.id() , "gbuff3"  );
	gbuffId[7]     = glGetUniformLocation( prLightsBase.id() , "gbuff4"  );
                                                                             
	atmId          = glGetUniformLocation( prAtmosphere.id() , "texture" );
	atmRadiusId    = glGetAttribLocation ( prAtmosphere.id() , "radius"  );
	atmAtmDataId   = glGetAttribLocation ( prAtmosphere.id() , "atmData" );
	atmAtmColorId  = glGetAttribLocation ( prAtmosphere.id() , "atmColor");
                                                                             
	gbuffId[8 ]    = glGetUniformLocation( prPostAtm.id() , "gbuff1"     );
	gbuffId[9 ]    = glGetUniformLocation( prPostAtm.id() , "gbuff2"     );
	gbuffId[10]    = glGetUniformLocation( prPostAtm.id() , "gbuff3"     );
	atmMaterialsId = glGetUniformLocation( prPostAtm.id() , "materials"  );
	atmModelId     = glGetAttribLocation ( prPostAtm.id() , "model"      );
	atmLightId     = glGetAttribLocation ( prPostAtm.id() , "light"      );

	prPlanet.use();
	glUniform1i( materialsTexId , 0 );
	glUniform1i( sphereTexId    , 1 );
	glUniform1i( normalsTexId   , 2 );
	glUniform1i( textureTexId   , 3 );
	prLighting.use();
	glUniform1i( gbuffId[0]     , 0 );
	glUniform1i( gbuffId[1]     , 1 );
	glUniform1i( gbuffId[2]     , 2 );
	glUniform1i( gbuffId[3]     , 3 );
	glUniform1i( matLId         , 4 );
	prLightsBase.use();         
	glUniform1i( gbuffId[4]     , 0 );
	glUniform1i( gbuffId[5]     , 1 );
	glUniform1i( gbuffId[6]     , 2 );
	glUniform1i( gbuffId[7]     , 3 );
	prAtmosphere.use();
	glUniform1i( atmId          , 0 );
	prPostAtm.use();
	glUniform1i( gbuffId[8]     , 0 );
	glUniform1i( gbuffId[9]     , 1 );
	glUniform1i( gbuffId[10]    , 2 );
	glUniform1i( atmMaterialsId , 3 );
	Program::none();

	create_textures( gfx->width() , gfx->height() );

	iftexturesId   = glGetUniformLocation( prPlanet.id() , "iftextures"  );
	ifnormalsId    = glGetUniformLocation( prPlanet.id() , "ifnormals"   );
	brightness     = glGetUniformLocation( prLightsBase.id(), "brightness");

//        tmptex = gfx->texMgr.loadTexture( DATA("glow_test.png") );
}

void DeferRender::resize( unsigned int width , unsigned int height )
{
	delete_textures();
	create_textures( width , height );
}

void DeferRender::update_configuration()
{
	gfx->cfg().get<bool>( "deffered.lighting" ) ?
		flags |= LIGHTING : flags &= ~LIGHTING;

	prPlanet.use();
	glUniform1i( iftexturesId , gfx->cfg().get<bool>( "deffered.textures" ) );
	glUniform1i( ifnormalsId  , gfx->cfg().get<bool>( "deffered.normals"  ) );

	prLightsBase.use();
	glUniform1f( brightness   , gfx->cfg().get<float>( "deffered.brightness" ) );

	prLighting.use();
	glUniform1i( ifplanesId   , gfx->cfg().get<bool>( "deffered.lights_range" ) );
	Program::none();
}

void DeferRender::on_camera_angle_changed( float*m )
{
	prPlanet.use();
	glUniformMatrix4fv( anglesId , 1 , GL_TRUE , m );
	Program::none();
}

void DeferRender::create_textures( unsigned int w , unsigned int h )
{
	unsigned sphereSize = pow(2,floor(log(std::max(w,h))/log(2.0)));
	sphereTex = generate_sphere_texture( sphereSize ,sphereSize );

	atmTex    = generate_atmosphere_texture( sphereSize * 1.5 , sphereSize );

	normalsTex= generate_normals_texture(sphereSize , sphereSize*2 );

	for( int i=0 ;i<gbuffNum ; i++ )
		gbuffTex[i] = generate_render_target_texture( w , h );

	screenTex   = generate_render_target_texture( w , h );

	glGenTextures( 1, &depthTex );
	glBindTexture( GL_TEXTURE_2D, depthTex );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, w, h, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0 );

//        generate_glow_planes( planes , glow_size , w );

	glGenFramebuffers( 3 , fboId );
	glBindFramebuffer( GL_FRAMEBUFFER , fboId[0] );

	bufferlist[0] = GL_COLOR_ATTACHMENT0;
	bufferlist[1] = GL_COLOR_ATTACHMENT1;
	bufferlist[2] = GL_COLOR_ATTACHMENT2;
	bufferlist[3] = GL_COLOR_ATTACHMENT3;

	glFramebufferTexture2D( GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 ,
				GL_TEXTURE_2D , gbuffTex[0] , 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT1 ,
				GL_TEXTURE_2D , gbuffTex[1] , 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT2 ,
				GL_TEXTURE_2D , gbuffTex[2] , 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT3 ,
				GL_TEXTURE_2D , gbuffTex[3] , 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER , GL_DEPTH_ATTACHMENT  ,
				GL_TEXTURE_2D , depthTex  , 0 );

//        glBindFramebuffer( GL_FRAMEBUFFER , fboId[1] );

//        glFramebufferTexture2D( GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 ,
//                                GL_TEXTURE_2D , screenTex , 0 );
//        glFramebufferTexture2D( GL_FRAMEBUFFER , GL_DEPTH_ATTACHMENT  ,
//                                GL_TEXTURE_2D , depthTex  , 0 );

	glBindFramebuffer( GL_FRAMEBUFFER , fboId[2] );

	glFramebufferTexture2D( GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 ,
				GL_TEXTURE_2D , screenTex , 0 );
	glBindFramebuffer( GL_FRAMEBUFFER , 0 );
}

void DeferRender::delete_textures()
{
	glDeleteTextures(1,&atmTex);
	glDeleteTextures(1,&sphereTex);
	glDeleteTextures(gbuffNum,gbuffTex);
	glDeleteTextures(1,&depthTex );
	glDeleteTextures(1,&screenTex);

	glDeleteTextures(1,&normalsTex);

	glDeleteFramebuffers( 3 , fboId );
}

GLuint DeferRender::generate_sphere_texture( int w , int h )
{
	float*sphere = new float[ w * h * 4 ];

	int w4 = w*4;
	float w2 = (float)w/2.0f;
	float h2 = (float)h/2.0f;

	for( int wi=0 ; wi<w4 ; wi+=4 )
		for( int hi=0 ; hi<h ; hi++ )
		{
			float x = ((float)wi/4.0f - w2)/(float)w2;
			float y = ((float)hi - h2)/(float)h2;
			float xxyy = x*x + y*y;
			float a = 1 <= xxyy ? 0.0 : 1.0 ;
			float z = (!a?0.0f:std::sqrt( 1 - xxyy ));

			int i = wi + hi*w4;

			sphere[ i     ] = x;
			sphere[ i + 1 ] = y;
			sphere[ i + 2 ] = z;
			sphere[ i + 3 ] = a;
		}

	GLuint texId;
	glGenTextures( 1 , &texId );
	glBindTexture(GL_TEXTURE_2D, texId );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S    , GL_CLAMP  );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T    , GL_CLAMP  );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,w,h,0,GL_RGBA,GL_FLOAT,sphere);
	delete[]sphere;
	return texId;
}

GLuint DeferRender::generate_atmosphere_texture( int w , int h )
{
	float*sphere = new float[ w * h * 4 ];

	int w4 = w*4;
	float w2 = (float)w/2.0f;
	float h2 = (float)h/2.0f;

	for( int wi=0 ; wi<w4 ; wi+=4 )
		for( int hi=0 ; hi<h ; hi++ )
		{
			float x = ((float)wi/4.0f - w2)/(float)w2;
			float y = ((float)hi - h2)/(float)h2;
			float xxyy = x*x + y*y;
			float a = 1 <= xxyy ? 0.0 : 1.0 ;
			float z = (!a?0.0f:std::sqrt( 1 - xxyy ));

			int i = wi + hi*w4;

			x /= sqrt( xxyy );
			y /= sqrt( xxyy );

			sphere[ i     ] = x;
			sphere[ i + 1 ] = y;
			sphere[ i + 2 ] = 0;
			sphere[ i + 3 ] = z;
		}

	GLuint texId;
	glGenTextures( 1 , &texId );
	glBindTexture(GL_TEXTURE_2D, texId );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S    , GL_CLAMP  );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T    , GL_CLAMP  );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,w,h,0,GL_RGBA,GL_FLOAT,sphere);
	delete[]sphere;
	return texId;
}

GLuint DeferRender::generate_render_target_texture( int w , int h )
{
	GLuint texId;
	glGenTextures( 1 , &texId );
	glBindTexture(GL_TEXTURE_2D, texId );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S    , GL_CLAMP  );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T    , GL_CLAMP  );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,w,h,0,GL_RGBA,GL_FLOAT,NULL);
	return texId;
}

GLuint DeferRender::generate_angles_texture( int w , int h )
{
	float* data = new float[w*h*3];

	memset( data , 0 , sizeof(float)*w*h*3 );

	int w3 = w*3;
	float w2 = (float)w/2.0f;
	float h2 = (float)h/2.0f;

	for( int hi = 0 ; hi<h ; hi++ )
		for( int wi = 0 ; wi<w3 ; wi+=3 )
		{
			float x = ((float)wi/3.0f - w2)/(float)w2;
			float y = ((float)hi - h2)/(float)h2;
			float xxyy = x*x + y*y;
			float a = 1 <= xxyy ? 0.0 : 1.0 ;
			float z = (!a?0.0f:std::sqrt( 1 - xxyy ));

			int i = hi*w3 + wi;

                        data[ i     ] =-atan2( z , x ) + PI/2.0f; // longitude
                        data[ i + 1 ] = asin( y );                // latitude
			data[ i + 2 ] = a;
		}

	GLuint texId;
	glGenTextures( 1 , &texId );
	glBindTexture(GL_TEXTURE_2D, texId );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S    , GL_CLAMP_TO_BORDER );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T    , GL_MIRRORED_REPEAT );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB16F,w,h,0,GL_RGB,GL_FLOAT,data);
	delete[]data;
	return texId;
}

GLuint DeferRender::generate_normals_texture( int w , int h )
{
	float* data = new float[w*h*3];

	memset(data,0,sizeof(float)*w*h*3);

	int w3 = w*3;
	float w2 = (float)w/2.0f;
	float h2 = (float)h/2.0f;

	for( int wi = 0 ; wi<w3 ; wi+=3 )
		for( int hi = 0 ; hi<h ; hi++ )
		{
			float xn = (((float)wi/3.0f/w2-1)*PI);
			float yn = (hi - h2) / (float)h * PI;

			// phi    - <-PI/2 , PI/2 >
			// lambda -   <-PI , PI >
			float phi    = yn;                 // latitude
			float lambda = xn / std::cos(phi); // longitude

//                        if( lambda < - PI || lambda > PI ) continue;

//                        log_printf(DBG,"phi %f , lmb %f\n",phi,lambda);

			float x =-sin( lambda ) * cos( phi );
			float y =-sin( phi );
			float z = cos( lambda ) * cos( phi );

//                        log_printf(DBG,"%f %f %f\n",x,y,z);

			int i = hi*w3 + wi;

			data[ i     ] = x;
			data[ i + 1 ] = y;
			data[ i + 2 ] = z;
		}

	GLuint texId;
	glGenTextures( 1 , &texId );
	glBindTexture(GL_TEXTURE_2D, texId );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S    , GL_CLAMP  );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T    , GL_MIRRORED_REPEAT );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB16F,w,h,0,GL_RGB,GL_FLOAT,data);
	delete[]data;
	return texId;
}

void DeferRender::generate_glow_planes( MEM::MISC::BufferGl<float>& buf , int num , int size )
{
	float ds = 1.0f/(float)size;

	buf.resize( (2+1+3)*(num*2+1)*4 );

	float*hf = buf.map( MEM::MISC::BUF_H );
	unsigned char*hub;
	for( int i=-num ; i<=num ; i++ )
	{
		float dx = ds*i;
		float dy = 0;
		unsigned char a = (unsigned char)(128.0f*(float)(num+1-abs(i))/(float)(num+1));

		*  hf = 0;          // texcoord.s
		*++hf = 0;          // texcoord.t
		 ++hf;
		hub = (unsigned char*)hf;
		*  hub= 255;
		*++hub= 255;
		*++hub= 255;
		*++hub= a;
		*++hf = -1.0f + dx; // vert.x
		*++hf = -1.0f + dy; // vert.y
		*++hf = 0;          // vert.z
		*++hf = 1;
		*++hf = 0;
		 ++hf;
		hub = (unsigned char*)hf;
		*  hub= 255;
		*++hub= 255;
		*++hub= 255;
		*++hub= a;
		*++hf =  1.0f + dx;
		*++hf = -1.0f + dy;
		*++hf = 0;
		*++hf = 1;
		*++hf = 1;
		 ++hf;
		hub = (unsigned char*)hf;
		*  hub= 255;
		*++hub= 255;
		*++hub= 255;
		*++hub= a;
		*++hf =  1.0f + dx;
		*++hf =  1.0f + dy;
		*++hf = 0;
		*++hf = 0;
		*++hf = 1;
		 ++hf;
		hub = (unsigned char*)hf;
		*  hub= 255;
		*++hub= 255;
		*++hub= 255;
		*++hub= a;
		*++hf = -1.0f + dx;
		*++hf =  1.0f + dy;
		*++hf = 0;
		 ++hf;
	}
	buf.unmap();
}

void DeferRender::draw() const
{
	glClearColor(0,0,0,0);
	glAlphaFunc( GL_GREATER , 0.1 );
	glEnable( GL_ALPHA_TEST );
	glEnable( GL_DEPTH_TEST );
	glDisable(GL_BLEND );

	glEnableClientState( GL_VERTEX_ARRAY );
	glEnableVertexAttribArray( radiusId );
	glEnableVertexAttribArray( modelId  );
	glEnableVertexAttribArray( texIdId  );
	glEnableVertexAttribArray( atmDataId );
	glEnableVertexAttribArray( atmColorId );

	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	factory->getPositions().unbind();

	factory->getRadiuses().bind();
	glVertexAttribPointer( radiusId , 1, GL_FLOAT, GL_FALSE, 0, NULL );
	factory->getRadiuses().unbind();

	factory->getModels().bind();
	glVertexAttribIPointer( modelId  , 1, GL_INT , 0, NULL );
	factory->getModels().unbind();

	factory->getTexIds().bind();
	glVertexAttribIPointer( texIdId  , 1, GL_INT , 0, NULL );
	factory->getTexIds().unbind();

	factory->getAtmData().bind();
	glVertexAttribPointer( atmDataId , 2, GL_FLOAT, GL_FALSE, 0, NULL );
	factory->getAtmData().unbind();

	factory->getAtmColor().bind();
	glVertexAttribPointer( atmColorId , 3, GL_FLOAT, GL_FALSE, 0, NULL );
	factory->getAtmColor().unbind();

	prPlanet.use();
	glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_1D, materialsTex );
	glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, sphereTex    );
	glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, normalsTex   );
	glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D_ARRAY, texturesTex );

	glBindFramebuffer( GL_FRAMEBUFFER , fboId[0] );
	glDrawBuffers( gbuffNum , bufferlist );

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDrawArrays( GL_POINTS , 0 , factory->size() );

	glBindFramebuffer( GL_FRAMEBUFFER , 0 );
	Program::none();
	glDisableVertexAttribArray( radiusId );
	glDisableVertexAttribArray( modelId  );
	glDisableVertexAttribArray( texIdId  );
	glDisableVertexAttribArray( atmDataId );
	glDisableVertexAttribArray( atmColorId );

	glDisable( GL_DEPTH_TEST );
	glEnable( GL_BLEND );
	glBlendFunc( GL_ONE , GL_ONE );

//        glClear( GL_DEPTH_BUFFER_BIT ); 

	glActiveTexture(GL_TEXTURE0); glBindTexture( GL_TEXTURE_2D, gbuffTex[0] );
	glActiveTexture(GL_TEXTURE1); glBindTexture( GL_TEXTURE_2D, gbuffTex[1] );
	glActiveTexture(GL_TEXTURE2); glBindTexture( GL_TEXTURE_2D, gbuffTex[2] );
	glActiveTexture(GL_TEXTURE3); glBindTexture( GL_TEXTURE_2D, gbuffTex[3] );

	glEnableVertexAttribArray( modelLId );
	glEnableVertexAttribArray( lightId );

	prLighting.use();

	glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_1D, materialsTex );

	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	factory->getPositions().unbind();

	factory->getModels().bind();
	glVertexAttribIPointer( modelLId  , 1, GL_INT , 0, NULL );
	factory->getModels().unbind();

	factory->getLight().bind();
	glVertexAttribPointer( lightId , 3 , GL_FLOAT , GL_FALSE , 0 , NULL );
	factory->getLight().unbind();

//        glBindFramebuffer( GL_FRAMEBUFFER , fboId[2] );
	glDrawBuffer( GL_COLOR_ATTACHMENT0 );
	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT);

	if( flags & LIGHTING )
		glDrawArrays( GL_POINTS , 0 , factory->size() );

	glDisableVertexAttribArray( lightId );
	glDisableVertexAttribArray( modelLId  );

	prLightsBase.use();
	glBegin(GL_POINTS);
	 glVertex3f(0,0,0);
	glEnd();

//        glBindFramebuffer( GL_FRAMEBUFFER , 0 );

	                                glBindTexture( GL_TEXTURE_1D , 0 );
	glActiveTexture( GL_TEXTURE3 ); glBindTexture( GL_TEXTURE_2D , 0 );
	glActiveTexture( GL_TEXTURE2 ); glBindTexture( GL_TEXTURE_2D , 0 );
	glActiveTexture( GL_TEXTURE1 ); glBindTexture( GL_TEXTURE_2D , 0 );
	glActiveTexture( GL_TEXTURE0 ); glBindTexture( GL_TEXTURE_2D , 0 );
	Program::none();

////        glEnable(GL_TEXTURE_2D);

//        glBlendFunc( GL_SRC_ALPHA , GL_ONE );

//        planes.bind();
//        glInterleavedArrays( GL_T2F_C4UB_V3F , 0 , NULL );
////        glVertexPointer  ( 2 , GL_FLOAT , sizeof(float)*2 , (const GLvoid*)0                 );
////        glTexCoordPointer( 2 , GL_FLOAT , sizeof(float)*2 , (const GLvoid*)(sizeof(float)*2) );
//        planes.unbind();

//        glEnable(GL_TEXTURE_2D);
//        tmptex->bind();

//        glDrawArrays( GL_QUADS , 0 , (glow_size*2+1)*4 );

//        Texture::unbind();

//        glBindFramebuffer( GL_READ_FRAMEBUFFER , fboId[2] );
//        glBlitFramebuffer(0,0,gfx->width(),gfx->height(),
//                          0,0,gfx->width(),gfx->height(),
//                          GL_COLOR_BUFFER_BIT,GL_NEAREST);
//        glBindFramebuffer( GL_READ_FRAMEBUFFER , 0 );

	glEnable( GL_DEPTH_TEST );
	glDisable( GL_BLEND );
//        glDisable( GL_DEPTH_TEST );
//        glBlendFunc( GL_SRC_ALPHA , GL_ONE_MINUS_SRC_ALPHA );

	glEnableVertexAttribArray( atmRadiusId );
	glEnableVertexAttribArray( atmAtmDataId );
	glEnableVertexAttribArray( atmAtmColorId );

	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	factory->getPositions().unbind();

	factory->getRadiuses().bind();
	glVertexAttribPointer( atmRadiusId , 1, GL_FLOAT, GL_FALSE, 0, NULL );
	factory->getRadiuses().unbind();

	factory->getAtmData().bind();
	glVertexAttribPointer( atmAtmDataId , 2, GL_FLOAT, GL_FALSE, 0, NULL );
	factory->getAtmData().unbind();

	factory->getAtmColor().bind();
	glVertexAttribPointer( atmAtmColorId , 3, GL_FLOAT, GL_FALSE, 0, NULL );
	factory->getAtmColor().unbind();

	prAtmosphere.use();

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D , atmTex );

	glBindFramebuffer( GL_FRAMEBUFFER , fboId[0] );
	glDrawBuffers( 3 , bufferlist );

	glClearColor(0,0,0,0);
	glClear( GL_COLOR_BUFFER_BIT );

	glDrawArrays( GL_POINTS , 0 , factory->size() );

	glBindFramebuffer( GL_FRAMEBUFFER , 0 );

	glBindTexture( GL_TEXTURE_2D , 0 );

	Program::none();

	glDisableVertexAttribArray( atmAtmColorId );
	glDisableVertexAttribArray( atmAtmDataId );
	glDisableVertexAttribArray( atmRadiusId );

	glDisable( GL_DEPTH_TEST );
	glEnable( GL_BLEND );
	glBlendFunc( GL_ONE , GL_ONE );

	glActiveTexture(GL_TEXTURE0); glBindTexture( GL_TEXTURE_2D, gbuffTex[0] );
	glActiveTexture(GL_TEXTURE1); glBindTexture( GL_TEXTURE_2D, gbuffTex[1] );
	glActiveTexture(GL_TEXTURE2); glBindTexture( GL_TEXTURE_2D, gbuffTex[2] );
	glActiveTexture(GL_TEXTURE3); glBindTexture( GL_TEXTURE_1D, materialsTex);

	glEnableVertexAttribArray( atmModelId );
	glEnableVertexAttribArray( atmLightId );

	prPostAtm.use();

	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	factory->getPositions().unbind();

	factory->getModels().bind();
	glVertexAttribIPointer( atmModelId  , 1, GL_INT , 0, NULL );
	factory->getModels().unbind();

	factory->getLight().bind();
	glVertexAttribPointer( atmLightId , 3 , GL_FLOAT , GL_FALSE , 0 , NULL );
	factory->getLight().unbind();

	if( flags & LIGHTING )
		glDrawArrays( GL_POINTS , 0 , factory->size() );

	Program::none();

	                                glBindTexture( GL_TEXTURE_1D , 0 );
	glActiveTexture( GL_TEXTURE2 ); glBindTexture( GL_TEXTURE_2D , 0 );
	glActiveTexture( GL_TEXTURE1 ); glBindTexture( GL_TEXTURE_2D , 0 );
	glActiveTexture( GL_TEXTURE0 ); glBindTexture( GL_TEXTURE_2D , 0 );

	glDisableVertexAttribArray( atmModelId );
	glDisableVertexAttribArray( atmLightId );

	glDisableClientState( GL_VERTEX_ARRAY );

//        glBlendFunc( GL_ONE_MINUS_DST_ALPHA  ,  GL_DST_ALPHA );
//        glEnable(GL_TEXTURE_2D);
//        glBindTexture( GL_TEXTURE_2D , screenTex );

//        glMatrixMode(GL_PROJECTION);
//        glPushMatrix();
//        glLoadIdentity();
//        glMatrixMode(GL_MODELVIEW);
//        glPushMatrix();
//        glLoadIdentity();
//        glBegin( GL_QUADS );
//          glTexCoord2f( 1 , 1 ); glVertex2f(  1 ,  1 );
//          glTexCoord2f( 0 , 1 ); glVertex2f( -1 ,  1 );
//          glTexCoord2f( 0 , 0 ); glVertex2f( -1 , -1 );
//          glTexCoord2f( 1 , 0 ); glVertex2f(  1 , -1 );
//        glEnd();
//        glPopMatrix();
//        glMatrixMode(GL_PROJECTION);
//        glPopMatrix();

//        glBindTexture( GL_TEXTURE_2D , 0 );

	glDisable( GL_ALPHA_TEST );
	glDisable( GL_BLEND );

//        glClear( GL_DEPTH_BUFFER_BIT );

	glBindFramebuffer( GL_READ_FRAMEBUFFER , fboId[0] );
	glBlitFramebuffer(0,0,gfx->width(),gfx->height(),
			  0,0,gfx->width(),gfx->height(),
			  GL_DEPTH_BUFFER_BIT,GL_NEAREST);
	glBindFramebuffer( GL_READ_FRAMEBUFFER , 0 );

	glDisable( GL_DEPTH_TEST );
}

