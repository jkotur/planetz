#include <GL/glew.h>
#include <SDL/SDL.h>

#include "./gfx.h"
#include "../util/logger.h"

#include "../constants.h"

using namespace GFX;

Gfx::~Gfx()
{
	SDL_Quit();
}

bool Gfx::SDL_init(int width,int height)
{
	log_printf(INFO,"Starting SDL...\n");

	if( SDL_Init(SDL_INIT_VIDEO|SDL_INIT_TIMER) ) {
		log_printf(CRITICAL,"SDL error occurred: %s\n",SDL_GetError());
		return false;
	}

	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1); 
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16); 
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);

	flags = SDL_OPENGL|SDL_HWSURFACE|SDL_DOUBLEBUF;
	flags |= SDL_RESIZABLE;
	if( FULLSCREEN_MODE )
		flags |= SDL_FULLSCREEN;

	reshape_window(width,height);

	return true;
}

bool Gfx::GL_init()
{
	log_printf(INFO,"Graphics init...\n");

	GLenum err = glewInit();
	if( GLEW_OK != err ) {
		log_printf(CRITICAL,"GLEW error: %s\n", glewGetErrorString(err));
		return false;
	}

	if( glewIsSupported("GL_VERSION_3_2") )
		log_printf(INFO,"[GL] Hurray! OpenGL 3.2 is supported.\n");
	else {
		log_printf(CRITICAL,"[GL] OpenGL 3.2 is not supported. Program cannot run corectly");
		return false;
	}

	GL_query();

	return GL_view_init();
}

void Gfx::GL_query()
{
	int ires;
	glGetIntegerv(GL_MAX_TEXTURE_SIZE,&ires);
	log_printf(INFO,"[GL] max texture size:             %d\n",ires);
	glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS,&ires);
	log_printf(INFO,"[GL] max array texture layers:     %d\n",ires);
	glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES,&ires);
	log_printf(INFO,"[GL] max geometry output vertices: %d\n",ires);
}

bool Gfx::GL_view_init()
{
	glViewport(0,0,mwidth,mheight);

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	int width=this->width() , height = this->height();

	float ambient [4] = { 0.3, 0.3, 0.3, 1.0 };
	float diffuse [4] = { 0.6, 0.6, 0.6, 1.0 };
	float specular[4] = { 0.7, 0.7, 0.7, 1.0 };
	float position[4] = { 0.0, 0.0, 10000.0, 1.0 };

	glShadeModel(GL_SMOOTH);

	glClearColor(0.0f,0.0f,0.0f,0.0f);
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glLightfv(GL_LIGHT0, GL_AMBIENT,  ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
	glLightfv(GL_LIGHT0, GL_POSITION, position);
//        glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION,  0.000000000f);
//        glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.00000000000000f);
//        glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION,  0.00000000000f);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);

	GL_viewport(mwidth,mheight);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluPerspective(75.0, (double)width/(double)height, 1, 10000);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	return true;
}

void Gfx::GL_viewport( int w , int h )
{
	glViewport(0,0,mwidth,mheight);
}

void Gfx::reshape_window(int width, int height)
{
	mwidth = width; mheight = height;
	if( !(drawContext = SDL_SetVideoMode(width,height, 0, flags)) ) {
		log_printf(CRITICAL,"Cannot set video mode: %s\n",SDL_GetError() );
		exit(-1);
	}

	height = height < 1 ? 1 : height;

	glViewport(0,0,width,height);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glPopAttrib();

	GL_viewport( width , height );
}

void Gfx::clear() const
{
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void Gfx::add( Drawable* d )
{ // FIXME: add prioritets
	d->setGfx( this );
	to_draw.push_back( d );
}

void Gfx::remove( Drawable* d )
{
	to_draw.remove(d);
}

void Gfx::render() const 
{
	clear();

	for( std::list<Drawable*>::const_iterator i = to_draw.begin() ; i!=to_draw.end() ; ++i )
		(*i)->draw();

	SDL_GL_SwapBuffers();
}

