#include "planetz_renderer.h"

#include <GL/glew.h>

using namespace GFX;

PlanetzRenderer::PlanetzRenderer( const GPU::GfxPlanetFactory * factory )
	: factory( factory )
{
}

PlanetzRenderer::~PlanetzRenderer()
{
}

void PlanetzRenderer::draw() const
{
	glPointSize( 3 );
	glEnableClientState( GL_VERTEX_ARRAY );
	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	glDrawArrays( GL_POINTS , 0 , factory->getPositions().getLen() );
	factory->getPositions().unbind();
	glDisableClientState( GL_VERTEX_ARRAY );
}

