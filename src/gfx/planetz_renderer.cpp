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
	pr.geomParams( GL_POINTS , GL_LINE_STRIP );

	pr.link();

#define size  1024

	float model[size*3];

	for( int i=0 ; i<size*3 ; i+=3 )
	{
		model[i  ] = (float)i/(float)(size*1.5);
		model[i+1] = (float)i/(float)(size*1.5);
		model[i+2] = (float)i/(float)(size*1.5);
	}

	//        glActiveTexture(GL_TEXTURE0);
	glGenTextures(1,&tex);
	glBindTexture(GL_TEXTURE_1D, tex);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage1D(GL_TEXTURE_1D,0,GL_RGB16F,size,0,GL_BGR, GL_FLOAT , model );

	modTexId = glGetUniformLocation( pr.id() , "models" );
}

void PlanetzRenderer::draw() const
{
	glPointSize( 3 );
	glEnableClientState( GL_VERTEX_ARRAY );

	//        glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_1D,tex);
	glUniform1i(modTexId,0);

	pr.use();
	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	glDrawArrays( GL_POINTS , 0 , factory->getPositions().getLen() );
	factory->getPositions().unbind();
	Program::none();

	glBindTexture(GL_TEXTURE_1D,0);

	glDisableClientState( GL_VERTEX_ARRAY );
}

