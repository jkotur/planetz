#include "planetz_tracer.h"

void GFX::PlanetsTracer::update()
{
	if( oldest >= number ) oldest = 0;

	positions.resize(number * gpf.getPositions().getLen());

	unsigned buffbytelen = gpf.getPositions().getLen()*3*sizeof(float);

	glBindBuffer( GL_COPY_READ_BUFFER , gpf.getPositions().getId() );
	glBindBuffer( GL_COPY_WRITE_BUFFER, positions.getId() );

	glCopyBufferSubData( GL_COPY_READ_BUFFER , GL_COPY_WRITE_BUFFER
	                   , 0 , buffbytelen*oldest
			   , buffbytelen );

	glBindBuffer( GL_COPY_READ_BUFFER , 0 );
	glBindBuffer( GL_COPY_WRITE_BUFFER, 0 );

	++oldest;
}

void GFX::PlanetsTracer::draw() const
{
	glEnableClientState( GL_VERTEX_ARRAY );

	glEnable( GL_DEPTH_TEST );

	glDepthFunc( GL_ALWAYS );

	glColor3f( 1 , 1 , 1 );
	glPointSize( 1.0f );

	positions.bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	positions.unbind();
	
	glDrawArrays( GL_POINTS , 0 , positions.getLen() );

	glDepthFunc( GL_LESS );

	glDisable( GL_DEPTH_TEST );

	glDisableClientState( GL_VERTEX_ARRAY );
}

