#include "planetz_renderer.h"

#include <GL/glew.h>

#include "constants.h"

using namespace GFX;

PlanetzRenderer::PlanetzRenderer( const GPU::GfxPlanetFactory * factory )
	: texModelId(0) , factory( factory ) 
{
}

PlanetzRenderer::~PlanetzRenderer()
{
	log_printf(DBG,"[DEL] Deleting PlanetzRenderer\n");
}

void PlanetzRenderer::setModels( GLuint tex )
{
	texModel = tex;
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
	pr.geomParams( GL_POINTS , GL_TRIANGLES );

	pr.link();

	texModelId = glGetUniformLocation( pr.id() , "models" );
}

void PlanetzRenderer::draw() const
{
	ASSERT_MSG( texModel , "Before drawing, models texture id must be specified by calling setModels" );

	glPointSize( 3 );
	glEnableClientState( GL_VERTEX_ARRAY );

	pr.use();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_1D,texModel);
	glUniform1i(texModelId,0);

	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	glDrawArrays( GL_POINTS , 0 , factory->getPositions().getLen() );
	factory->getPositions().unbind();
	Program::none();

	glBindTexture(GL_TEXTURE_1D,0);

	glDisableClientState( GL_VERTEX_ARRAY );
}

