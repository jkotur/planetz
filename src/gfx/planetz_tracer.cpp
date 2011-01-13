#include "planetz_tracer.h"

#include <cmath>

#include <boost/bind.hpp>

#include "gfx.h"
#include "util/logger.h"


void GFX::PlanetsTracer::start()
{
	if( dt <= 0 ) {
		log_printf(_WARNING,"Cannot update tracer so frequently (%f).\n",dt);
		return;
	}
	tc = timer.call(boost::bind(&GFX::PlanetsTracer::update,this),dt,true);
}

void GFX::PlanetsTracer::stop()
{
	tc.die();
}

void GFX::PlanetsTracer::clear()
{
	oldest= 0 ;
	begin = 0 ;
}

void GFX::PlanetsTracer::update_configuration()
{
	update_configuration( gfx->cfg() );
}

void GFX::PlanetsTracer::update_configuration( const Config& cfg )
{
	unsigned newnumber = cfg.get<unsigned>( "trace.length"    );
	double newdt       = cfg.get<double>  ( "trace.frequency" );
	drawable           = cfg.get<bool>    ( "trace.enable"    );

	if( newdt != dt ) {
		dt = newdt;
		if( tc.running() ) {
			stop();
			start();
		}
	}

	if( newnumber != number ) {
		number = newnumber;
		clear();
	}
}

void GFX::PlanetsTracer::update()
{
	if( oldest >= number ) oldest = 0;

	positions.resize(number * gpf.size());

	unsigned buffbytelen = gpf.size()*3*sizeof(float);

	glBindBuffer( GL_COPY_READ_BUFFER , gpf.getPositions().getId() );
	glBindBuffer( GL_COPY_WRITE_BUFFER, positions.getId() );

	glCopyBufferSubData( GL_COPY_READ_BUFFER , GL_COPY_WRITE_BUFFER
	                   , 0 , buffbytelen*oldest
			   , buffbytelen );

	glBindBuffer( GL_COPY_READ_BUFFER , 0 );
	glBindBuffer( GL_COPY_WRITE_BUFFER, 0 );

	++oldest;
	begin += gpf.size();
}

void GFX::PlanetsTracer::draw() const
{
	if( !drawable ) return;

	glEnableClientState( GL_VERTEX_ARRAY );

	glEnable( GL_DEPTH_TEST );

//        glDepthFunc( GL_ALWAYS );

	glColor3f( 1 , 1 , 1 );
	glPointSize( 1.0f );

	positions.bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	positions.unbind();
	
	glDrawArrays( GL_POINTS , 0 , std::min( positions.getLen() , begin ) );

//        glDepthFunc( GL_LESS );

	glDisable( GL_DEPTH_TEST );

	glDisableClientState( GL_VERTEX_ARRAY );
}

