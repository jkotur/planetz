#include <SDL/SDL.h>
#include <SDL/SDL_opengl.h>

#include "./gfx.h"
#include "../util/logger.h"

#include "../constants.h"

using namespace Gfx;

CGfx gfx;

// FIXME: all exlude SDL_init should be done in module functions

void CGfx::SDL_init(int width,int height)
{
	/*         SDL initialization*/
	int error;
	/*         initializing all subsystems (Timer, Audio, Video, CD, Joystick)*/
	error = SDL_Init(SDL_INIT_VIDEO|SDL_INIT_TIMER);

	if( error ) {
		log_printf(CRITICAL,"some SDL err mon\n");
		exit(1);
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
}

void CGfx::GL_init()
{
	log_printf(INFO,"Graphics init...\n");

	GL_view_init();
}

void CGfx::GL_view_init()
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

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluPerspective(75.0, (double)width/(double)height, 1, 10000);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	gluLookAt(
		0.0, 0.0, 10,//((double)width/2.0+200.0),
		0.0, 0.0, 0.0,                // View point (x,y,z)
		0.0, 1.0, 0.0                 // Up-vector (x,y,z)
	);
}

void CGfx::GL_viewport( int w , int h )
{
	glViewport(0,0,mwidth,mheight);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(75.0, (double)w/(double)h, 1, 10000);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
//        gluLookAt(
//                0.0, 0.0, 10,//((double)width/2.0+200.0),
//                0.0, 0.0, 0.0,                // View point (x,y,z)
//                0.0, 1.0, 0.0                 // Up-vector (x,y,z)
//        );
}

void CGfx::reshape_window(int width, int height)
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

	GL_view_init();
}

void CGfx::clear()
{
	glClear(  GL_DEPTH_BUFFER_BIT );

	float position[4] = { 0.0, 0.0, 100000.0, 1.0 };
	glLightfv(GL_LIGHT0, GL_POSITION, position);

	GL_viewport(mwidth,mheight);
}

