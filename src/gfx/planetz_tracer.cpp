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
	drawable           = cfg.get<bool>    ( "trace.visible"   );
	bool newtracing    = cfg.get<bool>    ( "trace.enable"    );

	if( newnumber != number ) {
		number = newnumber;
		clear();
	}

	if( !tracing && newtracing ) {
		positions = new MEM::MISC::BufferGl<float3>();
		clear();
		dt = newdt;
		if( !tc.running() ) start();
	} else if( tracing && !newtracing ) {
		delete positions;
		if( tc.running() ) stop();
	}
	tracing = newtracing;

	if( newdt != dt && tracing ) {
		dt = newdt;
		if( tc.running() ) {
			stop();
			start();
		}
	}
}

void GFX::PlanetsTracer::update()
{
	if( !tracing ) return;

	if( oldest >= number ) oldest = 0;

	positions->resize( number * gpf.size() , false );

	unsigned buffbytelen = gpf.size()*3*sizeof(float);

	glBindBuffer( GL_COPY_READ_BUFFER , gpf.getPositions().getId() );
	glBindBuffer( GL_COPY_WRITE_BUFFER, positions->getId() );

	glCopyBufferSubData( GL_COPY_READ_BUFFER , GL_COPY_WRITE_BUFFER
	                   , 0 , buffbytelen*oldest
			   , buffbytelen );

	glBindBuffer( GL_COPY_READ_BUFFER , 0 );
	glBindBuffer( GL_COPY_WRITE_BUFFER, 0 );

	++oldest;
	if( begin < positions->getLen() ) begin += gpf.size();
}

void GFX::PlanetsTracer::draw() const
{
	if( !drawable || !tracing ) return;

	glEnableClientState( GL_VERTEX_ARRAY );

	glEnable( GL_DEPTH_TEST );

//        glDepthFunc( GL_ALWAYS );

	glColor3f( 1 , 1 , 1 );
	glPointSize( 1.0f );

	positions->bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	positions->unbind();
	
	glDrawArrays( GL_POINTS , 0 , std::min( positions->getLen() , begin ) );

//        glDepthFunc( GL_LESS );

	glDisable( GL_DEPTH_TEST );

	glDisableClientState( GL_VERTEX_ARRAY );
}

