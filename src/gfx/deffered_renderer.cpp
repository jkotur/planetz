#include "deffered_renderer.h"

#include <GL/glew.h>
#include <cmath>
#include <algorithm>

#include "gfx.h"

#include "constants.h"

using namespace GFX;

DeferRender::DeferRender( const MEM::MISC::GfxPlanetFactory * factory )
	: materialsTex(0) , factory(factory) , flags(NULL)
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

	texture = gfx->texMgr.loadTexture(DATA("textures/mars.jpg"));

	sphereTexId    = glGetUniformLocation( prPlanet.id() , "sph_pos"   );
	materialsTexId = glGetUniformLocation( prPlanet.id() , "materialsTex" );

	anglesTexId    = glGetUniformLocation( prPlanet.id() , "anglesTex" );
	normalsTexId   = glGetUniformLocation( prPlanet.id() , "normalsTex");
	textureTexId   = glGetUniformLocation( prPlanet.id() , "textureTex");

	anglesId       = glGetUniformLocation( prPlanet.id() , "angles"    );
                                                                          
	radiusId       = glGetAttribLocation ( prPlanet.id() , "radius"    );
	modelId        = glGetAttribLocation ( prPlanet.id() , "model"     );

	iftexturesId   = glGetUniformLocation( prPlanet.id() , "textures"  );

	prPlanet.use();
	glUniform1i( materialsTexId , 0 );
	glUniform1i( sphereTexId    , 1 );
//        glUniform1i( anglesTexId    , 1 );
	glUniform1i( normalsTexId   , 2 );
	glUniform1i( textureTexId   , 3 );
	Program::none();

	gbuffId[0] = glGetUniformLocation( prLighting.id() , "gbuff1"   );
	gbuffId[1] = glGetUniformLocation( prLighting.id() , "gbuff2"   );
	gbuffId[2] = glGetUniformLocation( prLighting.id() , "gbuff3"   );
	gbuffId[3] = glGetUniformLocation( prLighting.id() , "gbuff4"   );
	matLId     = glGetUniformLocation( prLighting.id() , "materials");

	ifplanesId = glGetUniformLocation( prLighting.id() , "planes"   );

	modelLId   = glGetAttribLocation( prLighting.id()  , "model"    );
	emissiveLId= glGetAttribLocation( prLighting.id()  , "emissive" );

	gbuffId[4] = glGetUniformLocation( prLightsBase.id() , "gbuff1" );
	gbuffId[5] = glGetUniformLocation( prLightsBase.id() , "gbuff2" );
	gbuffId[6] = glGetUniformLocation( prLightsBase.id() , "gbuff3" );
	gbuffId[7] = glGetUniformLocation( prLightsBase.id() , "gbuff4" );

	prLighting.use();
	glUniform1i( gbuffId[0] , 0 );
	glUniform1i( gbuffId[1] , 1 );
	glUniform1i( gbuffId[2] , 2 );
	glUniform1i( gbuffId[3] , 3 );
	glUniform1i( matLId     , 4 );
	Program::none();

	prLightsBase.use();
	glUniform1i( gbuffId[4] , 0 );
	glUniform1i( gbuffId[5] , 1 );
	glUniform1i( gbuffId[6] , 2 );
	glUniform1i( gbuffId[7] , 3 );
	Program::none();

	create_textures( gfx->width() , gfx->height() );
}

void DeferRender::resize( unsigned int width , unsigned int height )
{
	delete_textures();
	create_textures( width , height );
}

void DeferRender::update_configuration()
{
	gfx->cfg().get<bool>( "lighting" ) ?
		flags |= LIGHTING : flags &= ~LIGHTING;

	prPlanet.use();
	glUniform1i( iftexturesId , gfx->cfg().get<bool>( "textures" ) );
	Program::none();

	prLighting.use();
	glUniform1i( ifplanesId   , gfx->cfg().get<bool>( "lightsplanes" ) );
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

	anglesTex = generate_angles_texture( sphereSize , sphereSize );
	normalsTex= generate_normals_texture(sphereSize , sphereSize*2 );

	for( int i=0 ;i<gbuffNum ; i++ )
		gbuffTex[i] = generate_render_target_texture( w , h );

	glGenTextures( 1, &depthTex );
	glBindTexture( GL_TEXTURE_2D, depthTex );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, w, h, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0 );

	glGenFramebuffers( 1, &fboId );
	glBindFramebuffer( GL_FRAMEBUFFER , fboId );

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

	glBindFramebuffer( GL_FRAMEBUFFER , 0 );

}

void DeferRender::delete_textures()
{
	glDeleteTextures(1,&sphereTex);
	glDeleteTextures(gbuffNum,gbuffTex);
	glDeleteTextures(1,&depthTex );

	glDeleteTextures(1,&anglesTex);
	glDeleteTextures(1,&normalsTex);

	glDeleteFramebuffers( 1 , &fboId );
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

void DeferRender::draw() const
{
	glAlphaFunc( GL_GREATER, 0.1 );
	glEnable( GL_ALPHA_TEST );

	glEnableClientState( GL_VERTEX_ARRAY );
	glEnableVertexAttribArray( radiusId );
	glEnableVertexAttribArray( modelId  );

	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	factory->getPositions().unbind();

	factory->getRadiuses().bind();
	glVertexAttribPointer( radiusId , 1, GL_FLOAT, GL_FALSE, 0, NULL );
	factory->getRadiuses().unbind();

	factory->getModels().bind();
	glVertexAttribIPointer( modelId  , 1, GL_INT , 0, NULL );
	factory->getModels().unbind();

	prPlanet.use();
	glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_1D, materialsTex );
	glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, sphereTex    );
//        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, anglesTex    );
	glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, normalsTex   );
	glActiveTexture(GL_TEXTURE3); texture->bind();

	glBindFramebuffer( GL_FRAMEBUFFER , fboId );

	glDrawBuffers( gbuffNum , bufferlist );

	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDrawArrays( GL_POINTS , 0 , factory->getPositions().getLen() );

	Program::none();
	glBindFramebuffer( GL_FRAMEBUFFER , 0 );
	glDisableVertexAttribArray( radiusId );
	glDisableVertexAttribArray( modelId  );

//        glDisable(GL_DEPTH_TEST);

	glClear( GL_DEPTH_BUFFER_BIT ); 

	glActiveTexture(GL_TEXTURE0); glBindTexture( GL_TEXTURE_2D, gbuffTex[0] );
	glActiveTexture(GL_TEXTURE1); glBindTexture( GL_TEXTURE_2D, gbuffTex[1] );
	glActiveTexture(GL_TEXTURE2); glBindTexture( GL_TEXTURE_2D, gbuffTex[2] );
	glActiveTexture(GL_TEXTURE3); glBindTexture( GL_TEXTURE_2D, gbuffTex[3] );

	prLightsBase.use();
	glBegin(GL_POINTS);
	 glVertex3f(0,0,0);
	glEnd();
	Program::none();

	glEnableVertexAttribArray( modelLId );
	glEnableVertexAttribArray( emissiveLId );

	glEnable( GL_BLEND );
	glBlendFunc( GL_ONE , GL_ONE );
//        glBlendFunc( GL_SRC_ALPHA , GL_ONE_MINUS_SRC_ALPHA );
//        glDisable( GL_DEPTH_TEST );

	prLighting.use();

	glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_1D, materialsTex );

	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	factory->getPositions().unbind();

	factory->getModels().bind();
	glVertexAttribIPointer( modelLId  , 1, GL_INT , 0, NULL );
	factory->getModels().unbind();

	factory->getEmissive().bind();
	glVertexAttribPointer( emissiveLId , 1 , GL_FLOAT , GL_FALSE , 0 , NULL );
	factory->getEmissive().unbind();

	if( flags & LIGHTING )
		glDrawArrays( GL_POINTS , 0 , factory->getPositions().getLen() );

	                                glBindTexture( GL_TEXTURE_1D , 0 );
	glActiveTexture( GL_TEXTURE3 ); glBindTexture( GL_TEXTURE_2D , 0 );
	glActiveTexture( GL_TEXTURE2 ); glBindTexture( GL_TEXTURE_2D , 0 );
	glActiveTexture( GL_TEXTURE1 ); glBindTexture( GL_TEXTURE_2D , 0 );
	glActiveTexture( GL_TEXTURE0 ); glBindTexture( GL_TEXTURE_2D , 0 );
	Program::none();

//        glEnable(GL_DEPTH_TEST);
	glDisable( GL_BLEND );
	glDisable( GL_ALPHA_TEST );

	glDisableVertexAttribArray( emissiveLId );
	glDisableVertexAttribArray( modelLId  );
	glDisableClientState( GL_VERTEX_ARRAY );
}

