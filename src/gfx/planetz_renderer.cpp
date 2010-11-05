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
//        pr.attach( gs );

//        glProgramParameteriEXT(pr.id(),GL_GEOMETRY_INPUT_TYPE_EXT,GL_POINTS);
//        glProgramParameteriEXT(pr.id(),GL_GEOMETRY_OUTPUT_TYPE_EXT,GL_POINTS);

	pr.link();
}

void PlanetzRenderer::draw() const
{
	glPointSize( 3 );
	glEnableClientState( GL_VERTEX_ARRAY );
	pr.use();
	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	glDrawArrays( GL_POINTS , 0 , factory->getPositions().getLen() );
	factory->getPositions().unbind();
	Program::none();
	glDisableClientState( GL_VERTEX_ARRAY );
}

