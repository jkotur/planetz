#include "deffered_renderer.h"

#include <GL/glew.h>
#include <cmath>

#include "gfx.h"

#include "constants.h"

using namespace GFX;

DeferRender::DeferRender( const MEM::MISC::GfxPlanetFactory * factory )
	: factory(factory)
{
	TODO("Connect this class to OnWindowResize event so it can resize its buffers");
	TODO("Move ShaderManager to gfx class");

	ShaderManager shm;

	prPlanet.create(
		shm.loadShader(GL_VERTEX_SHADER  ,DATA("shaders/deffered_01.vert")),
		shm.loadShader(GL_FRAGMENT_SHADER,DATA("shaders/deffered_01.frag")),
		shm.loadShader(GL_GEOMETRY_SHADER,DATA("shaders/deffered_01.geom")),
		GL_POINTS , GL_QUAD_STRIP );

	prLighting.create(
		shm.loadShader(GL_VERTEX_SHADER  ,DATA("shaders/deffered_02.vert")),
		shm.loadShader(GL_FRAGMENT_SHADER,DATA("shaders/deffered_02.frag")));

	radiusId = glGetAttribLocation( prPlanet.id() , "radius" );
	modelId  = glGetAttribLocation( prPlanet.id() , "model"  );

	TODO("Dynamically change screen size ratio in shader");
	GLint ratioId = glGetUniformLocation( prPlanet.id() , "ratio" );

	prPlanet.use();
	glUniform1i( sphereTexId , 0 );
	glUniform1f( ratioId , (float)BASE_W/(float)BASE_H );
	Program::none();

	sphereTexId = glGetUniformLocation( prPlanet.id() , "sph_pos" );
	sphereTex = generate_sphere_texture( 512 , 512 );

	for( int i=0 ;i<gbuffNum ; i++ )
		gbuffTex[i] = generate_render_target_texture( 800 , 600 );

	glGenTextures( 1, &depthTex );
	glBindTexture( GL_TEXTURE_2D, depthTex );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 800, 600, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0 );

	glGenFramebuffers( 1, &fboId );
	glBindFramebuffer( GL_FRAMEBUFFER , fboId );

	bufferlist[0] = GL_COLOR_ATTACHMENT0;
	bufferlist[1] = GL_COLOR_ATTACHMENT1;

	glFramebufferTexture2D( GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 , GL_TEXTURE_2D , gbuffTex[0] , 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT1 , GL_TEXTURE_2D , gbuffTex[1] , 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER , GL_DEPTH_ATTACHMENT  , GL_TEXTURE_2D , depthTex  , 0 );

	glBindFramebuffer( GL_FRAMEBUFFER , 0 );

	rect.resize( 8 );
	float * hptr = rect.map( MEM::MISC::BUF_H );
	hptr[0] = 0.0; hptr[1] = 0.0;
	hptr[2] = 1.0; hptr[3] = 0.0;
	hptr[4] = 0.0; hptr[5] = 1.0;
	hptr[6] = 1.0; hptr[7] = 1.0;
	rect.unmap();

	gbuffId[0] = glGetUniformLocation( prLighting.id() , "gbuff1" );
	gbuffId[1] = glGetUniformLocation( prLighting.id() , "gbuff2" );

	prLighting.use();
	glUniform1i( gbuffId[0] , 0 );
	glUniform1i( gbuffId[1] , 1 );
	Program::none();
}

DeferRender::~DeferRender()
{
	glDeleteTextures(1,&sphereTex);
	glDeleteTextures(gbuffNum,gbuffTex);
	glDeleteTextures(1,&depthTex );

	glDeleteFramebuffers( 1 , &fboId );
}

void DeferRender::prepare()
{
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
			float a = 1 <= x*x + y*y ? 0.0 : 1.0 ;
			float z = !a?0.0f:std::sqrt( 1 - x*x - y*y ); 

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
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,w,h,0,GL_RGBA,GL_FLOAT,NULL);
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
	glVertexAttribPointer( modelId  , 1, GL_FLOAT , GL_FALSE, 0, NULL );
	factory->getModels().unbind();

	prPlanet.use();
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D , sphereTex );

	glBindFramebuffer( GL_FRAMEBUFFER , fboId );

	glDrawBuffers( gbuffNum , bufferlist );

	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDrawArrays( GL_POINTS , 0 , factory->getPositions().getLen() );

	Program::none();
	glBindFramebuffer( GL_FRAMEBUFFER , 0 );
	glDisableVertexAttribArray( radiusId );
	glDisableVertexAttribArray( modelId  );

	glEnableClientState( GL_TEXTURE_COORD_ARRAY );

	glClear( GL_DEPTH_BUFFER_BIT ); 

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho( 0 , 1 , 0 , 1 , 1 , -1 );
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	prLighting.use();
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D , gbuffTex[0] );

	glActiveTexture( GL_TEXTURE1 );
	glBindTexture( GL_TEXTURE_2D , gbuffTex[1] );

	rect.bind();
	glTexCoordPointer( 2 , GL_FLOAT , 0 , NULL );
	glVertexPointer  ( 2 , GL_FLOAT , 0 , NULL );
	rect.unbind();
	glDrawArrays( GL_TRIANGLE_STRIP  , 0 , 4 ); 

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);

	glBindTexture( GL_TEXTURE_2D , 0 );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D , 0 );
	Program::none();

	glDisable( GL_ALPHA_TEST );
	glDisable( GL_BLEND );

	glDisableClientState( GL_TEXTURE_COORD_ARRAY );
	glDisableClientState( GL_VERTEX_ARRAY );
}

