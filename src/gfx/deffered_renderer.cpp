#include "deffered_renderer.h"

#include <GL/glew.h>

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

	GLint ratioId = glGetUniformLocation( pr.id() , "ratio" );

	TODO("Dynamically change screen size ratio in shader");

	pr.link();

	pr.use();
	glUniform1f( ratioId , (float)BASE_W/(float)BASE_H );
	Program::none();
}

DeferRender::~DeferRender()
{
}

void DeferRender::draw() const
{
	glEnableClientState( GL_VERTEX_ARRAY );

	glActiveTexture(GL_TEXTURE0);

	pr.use();
	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	glDrawArrays( GL_POINTS , 0 , factory->getPositions().getLen() );
	factory->getPositions().unbind();
	Program::none();

	glBindTexture(GL_TEXTURE_1D,0);

	glDisableClientState( GL_VERTEX_ARRAY );
}

