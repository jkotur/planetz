#include "window.h"

#include <GL/glew.h>

#include "constants.h"

#include "util/logger.h"

Window::Window( unsigned int w , unsigned int h )
	: w(w) , h(h) , err(0)
{
	if( !SDL_init(w,h) ) {
		err = 1;
		return;
	}
	if( !GL_init() ) {
		err = 2;
		return;
	}
}

bool Window::SDL_init( unsigned int w , unsigned int h )
{
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
	flags|= SDL_RESIZABLE;
	if( FULLSCREEN_MODE )
		flags |= SDL_FULLSCREEN;

	reshape_window( w , h );

	return true;
}

bool Window::GL_init()
{
	log_printf(INFO,"Graphics init...\n");

	GLenum err = glewInit();
	if( GLEW_OK != err ) {
		log_printf(CRITICAL,"[GLEW] %s\n", glewGetErrorString(err));
		return false;
	}

	if( glewIsSupported("GL_VERSION_3_2") )
		log_printf(INFO,"[GL] Hurray! OpenGL 3.2 is supported.\n");
	else {
		log_printf(CRITICAL,"[GL] OpenGL 3.2 is not supported. Program cannot run corectly\n");
		return false;
	}

	GL_query();

	return true;
}

void Window::GL_query()
{
	int ires;
	glGetIntegerv(GL_MAX_TEXTURE_SIZE,&ires);
	log_printf(INFO,"[GL] max texture size:             %d\n",ires);
	glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS,&ires);
	log_printf(INFO,"[GL] max array texture layers:     %d\n",ires);
	glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES,&ires);
	log_printf(INFO,"[GL] max geometry output vertices: %d\n",ires);
	glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS,&ires);
	log_printf(INFO,"[GL] texture units:                %d\n",ires);
	glGetIntegerv(GL_MAX_DRAW_BUFFERS,&ires);
	log_printf(INFO,"[GL] max draw buffers:             %d\n",ires);
}

Window::~Window()
{
	SDL_Quit();
	log_printf(DBG,"[DEL] SDL_Quit\n");
}

void Window::reshape_window( unsigned int _w , unsigned int _h )
{
	w = _w; h = _h;
	if( !(drawContext = SDL_SetVideoMode(w,h, 0, flags)) ) {
		log_printf(CRITICAL,"Cannot set video mode: %s\n",SDL_GetError() );
		exit(3);
	}
}

