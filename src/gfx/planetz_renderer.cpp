#include "planetz_renderer.h"

#include <GL/glew.h>

#include "constants.h"

using namespace GFX;

PlanetzRenderer::PlanetzRenderer( const GPU::GfxPlanetFactory * factory )
	: factory( factory )
{
}

PlanetzRenderer::~PlanetzRenderer()
{
}

void PlanetzRenderer::prepare()
{
	ShaderManager shm;

	Shader*vs = shm.loadShader(GL_VERTEX_SHADER,DATA("shaders/shader.vert"));
	Shader*fs = shm.loadShader(GL_FRAGMENT_SHADER,DATA("shaders/shader.frag"));
	Shader*gs = shm.loadShader(GL_GEOMETRY_SHADER,DATA("shaders/shader.geom"));

	pr.attach( vs );
	pr.attach( fs );
	pr.attach( gs );
	pr.geomParams( GL_POINTS , GL_TRIANGLE_STRIP );

	pr.link();

#define size 8

	float model[size*3] = { 
		 1 , 1 , 1 ,
		 1 ,-1 , 1 ,
		 1 ,-1 ,-1 ,
		 1 , 1 ,-1 ,
		-1 , 1 ,-1 ,
		-1 , 1 , 1 ,
		-1 ,-1 , 1 ,
		-1 ,-1 ,-1 };

	glGenTextures(3,tex);
	glBindTexture(GL_TEXTURE_1D, tex[0]);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage1D(GL_TEXTURE_1D,0,GL_RGB16F,size,0,GL_RGB, GL_FLOAT , model );

	modTexId = glGetUniformLocation( pr.id() , "models" );
	numId    = glGetUniformLocation( pr.id() , "num");
}

void PlanetzRenderer::draw() const
{
	glPointSize( 3 );
	glEnableClientState( GL_VERTEX_ARRAY );

	pr.use();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_1D,tex[0]);
	glUniform1i(modTexId,0);

	glUniform1i(numId,size);

	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	glDrawArrays( GL_POINTS , 0 , factory->getPositions().getLen() );
	factory->getPositions().unbind();
	Program::none();

	glBindTexture(GL_TEXTURE_1D,0);

	glDisableClientState( GL_VERTEX_ARRAY );
}

