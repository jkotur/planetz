#include "deffered_renderer.h"

#include <GL/glew.h>
#include <cmath>

#include "constants.h"

using namespace GFX;

DeferRender::DeferRender( const GPU::GfxPlanetFactory * factory )
	: factory(factory)
{
	ShaderManager shm;

	Shader*vs = shm.loadShader(GL_VERTEX_SHADER,DATA("shaders/deffered.vert"));
	Shader*fs = shm.loadShader(GL_FRAGMENT_SHADER,DATA("shaders/deffered.frag"));
	Shader*gs = shm.loadShader(GL_GEOMETRY_SHADER,DATA("shaders/deffered.geom"));

	pr.attach( vs );
	pr.attach( fs );
	pr.attach( gs );
	pr.geomParams( GL_POINTS , GL_QUAD_STRIP );

	pr.link();

	radiusId = glGetAttribLocation( pr.id() , "in_radiuses" );

	TODO("Dynamically change screen size ratio in shader");
	GLint ratioId = glGetUniformLocation( pr.id() , "ratio" );

	pr.use();
	glUniform1f( ratioId , (float)BASE_W/(float)BASE_H );
	Program::none();

	sphereTexId = glGetUniformLocation( pr.id() , "sph_pos" );
	sphereTex = generate_sphere_texture( 512 , 512 );
}

DeferRender::~DeferRender()
{
	glDeleteTextures(1,&sphereTex);
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
			float x = ((float)wi/4.0f - w2)/(float)w;
			float y = ((float)hi - h2)/(float)h;
			float a = .25 <= x*x + y*y ? 0.0 : 1.0 ;
			float z = !a?0.0f:std::sqrt( .25 - x*x - y*y ); 

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
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,w,h,0,GL_RGBA,GL_FLOAT,sphere);
	delete[]sphere;

	return texId;
}

void DeferRender::draw() const
{
	glAlphaFunc( GL_GREATER, 0.1 );
	glEnable( GL_ALPHA_TEST );

	glEnableClientState( GL_VERTEX_ARRAY );
	glEnableVertexAttribArray( radiusId );

	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	factory->getPositions().unbind();

	factory->getRadiuses().bind();
	glVertexAttribPointer( radiusId , 1, GL_FLOAT, GL_FALSE, 0, NULL );
	factory->getRadiuses().unbind();

	pr.use();
	glBindTexture( GL_TEXTURE_2D , sphereTex );
	glUniform1i( sphereTexId , 0 );

	glDrawArrays( GL_POINTS , 0 , factory->getPositions().getLen() );
	Program::none();

	glDisableClientState( GL_VERTEX_ARRAY );
	glDisableVertexAttribArray( radiusId );

	glDisable( GL_ALPHA_TEST );
}

