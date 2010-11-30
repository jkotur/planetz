#include "gfx.h"

#include <GL/glew.h>
#include <SDL/SDL.h>

#include <utility>

#include <boost/foreach.hpp>

#include "util/logger.h"
#include "constants.h"
#include "debug/routines.h"

using namespace GFX;

Gfx::~Gfx()
{
	log_printf(DBG,"[DEL] Deleting Gfx\n");
}

bool Gfx::window_init(int width,int height)
{
	mwidth = width; mheight = height;

	log_printf(INFO,"Starting window...\n");

	reshape_window(width,height);

	return GL_view_init();
}

bool Gfx::GL_view_init()
{
	glViewport(0,0,mwidth,mheight);

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glShadeModel(GL_SMOOTH);

	glClearColor(0.0f,0.0f,0.0f,0.0f);
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	float ambient [4] = { 0.3, 0.3, 0.3, 1.0 };
	float diffuse [4] = { 0.6, 0.6, 0.6, 1.0 };
	float specular[4] = { 0.7, 0.7, 0.7, 1.0 };
	float position[4] = { 0.0, 0.0, 10000.0, 1.0 };

	glLightfv(GL_LIGHT0, GL_AMBIENT,  ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
	glLightfv(GL_LIGHT0, GL_POSITION, position);
//        glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION,  0.000000000f);
//        glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.00000000000000f);
//        glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION,  0.00000000000f);
	glEnable(GL_LIGHT0);

	GL_viewport(mwidth,mheight);

	return true;
}

void Gfx::GL_viewport( int w , int h )
{
	glViewport(0,0,w,h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(75.0, (double)w/(double)h, 1, 10000);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void Gfx::reshape_window(int width, int height)
{
	mwidth = width; mheight = height;

	height = height < 1 ? 1 : height;

	GL_viewport( width , height );

	typedef std::pair<int,Drawable*> pair;
	BOOST_FOREACH( pair i , to_draw )
		i.second->resize( width , height );
}

void Gfx::clear() const
{
	glViewport(0,0,mwidth,mheight);

	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void Gfx::add( Drawable* d , int prior )
{ 
	d->setGfx( this );
	for( std::list<std::pair<int,Drawable*> >::iterator i = to_draw.begin() ;
	     ; ++i )
		if( i->first > prior || i==to_draw.end() ) {
			to_draw.insert( i , std::make_pair( prior , d ) );
			break;
		}
}

void Gfx::remove( Drawable* d )
{
	for( std::list<std::pair<int,Drawable*> >::iterator i = to_draw.begin() ;
	     i != to_draw.end() ; )
		if( i->second == d )
			i = to_draw.erase( i );
		else	++i;
}

void Gfx::render() const 
{
	clear();

	for( std::list<std::pair<int,Drawable*> >::const_iterator i = to_draw.begin() ; i!=to_draw.end() ; ++i )
		i->second->draw();

	SDL_GL_SwapBuffers();
}

