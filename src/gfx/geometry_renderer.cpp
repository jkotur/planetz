#include "geometry_renderer.h"

#include "constants.h"

using namespace GFX;

GeomRender::GeomRender( const GPU::GfxPlanetFactory * factory )
	: texModelId(0) , factory(factory)
{
	ShaderManager shm;

	Shader*vs = shm.loadShader(GL_VERTEX_SHADER,DATA("shaders/geometry.vert"));
	Shader*fs = shm.loadShader(GL_FRAGMENT_SHADER,DATA("shaders/geometry.frag"));
	Shader*gs = shm.loadShader(GL_GEOMETRY_SHADER,DATA("shaders/geometry.geom"));

	pr.attach( vs );
	pr.attach( fs );
	pr.attach( gs );
	pr.geomParams( GL_TRIANGLES , GL_TRIANGLE_STRIP );

	pr.link();

	texModelId = glGetUniformLocation( pr.id() , "models" );

}

GeomRender::~GeomRender()
{
}

void GeomRender::setModels( GPU::PlanetzModel mod )
{
	modPlanet = mod;
}

void GeomRender::draw() const
{
	ASSERT_MSG( modPlanet.vertices , "Before drawing, models texture id must be specified by calling setModels" );

	glPointSize( 3 );
	glEnableClientState( GL_VERTEX_ARRAY );

	glActiveTexture(GL_TEXTURE0);

	pr.use();
	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );

	for( int i=0 ; i<modPlanet.parts ; i++ )
	{
		glBindTexture(GL_TEXTURE_1D,modPlanet.vertices[i]);
		glUniform1i(texModelId,0);

		glDrawArrays( GL_POINTS , 0 , factory->getPositions().getLen() );
	}

	factory->getPositions().unbind();
	Program::none();

	glBindTexture(GL_TEXTURE_1D,0);

	glDisableClientState( GL_VERTEX_ARRAY );
}

