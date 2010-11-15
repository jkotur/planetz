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

	pr.link();

	radiusId = glGetAttribLocation( pr.id() , "in_radiuses" );

	TODO("Dynamically change screen size ratio in shader");
	GLint ratioId = glGetUniformLocation( pr.id() , "ratio" );

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
	glEnableVertexAttribArray( radiusId );

	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	factory->getPositions().unbind();

	factory->getRadiuses().bind();
	glVertexAttribPointer( radiusId , 1, GL_FLOAT, GL_FALSE, 0, NULL );
	factory->getRadiuses().unbind();

	pr.use();
	glDrawArrays( GL_POINTS , 0 , factory->getPositions().getLen() );
	Program::none();

	glDisableClientState( GL_VERTEX_ARRAY );
	glDisableVertexAttribArray( radiusId );
}

