#ifdef _WIN32
# include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <SDL/SDL.h>
#include <SDL/SDL_opengl.h>

#include <boost/bind.hpp>

#include "./util/timer/timer.h"
#include "./util/logger.h"
#include "./util/animation.h"

#include "./gfx/gfx.h"
#include "./gfx/background.h"

#include "./gfx/planet.h"
#include "./phx/planet.h"
#include "./planet.h"
#include "./planetz_manager.h"

#include "./saver.h"

#include "./constants.h"

#include "ui/ui.h"

void do_fps()
{
	static double oldtime = timer.get_dt_s();
	static int fps = 0;

	if( timer.get() - oldtime > 1 ) {
		oldtime = timer.get();
		log_printf(INFO,"fps: %d\n",fps);
		//                Gfx::Hud::fps = fps;
		fps = 0;
	}
	fps++;
}

#define BUFSIZE 512

#define CAM_START_VECS Vector3(0,0,10),Vector3(0,0,0),Vector3(0,1,0)

bool anim_pause = true;
void pause_toggle()
{
	anim_pause = !anim_pause;
}

void pause_anim()
{	anim_pause = true; }

void reset( Planetz*pl , Camera*c )
{
	pl->clear();
	c->set_perspective(CAM_START_VECS);
	anim_pause = true;
	pl->select(-1); // clear selection
}

int processHits (GLint hits, GLuint buffer[])
{
	int i;
	unsigned int j;
	GLuint names, *ptr;
	int outn = -1;
	double outz = 100.0;

	ptr = (GLuint *) buffer;                                            
	for (i = 0; i < hits; i++) {  /* for each hit  */
		names = *ptr;
		ptr++; // skip numbers
		if( outz <= (float)*ptr/0x7fffffff) { ptr+=2+names; continue; }
		outz = (float)*ptr/0x7fffffff;
		ptr++; // skip z1
		ptr++; // skip z2
		for (j = 0; j < names; j++) {  /* for each name */
			outn = *ptr;
			ptr++;
		}
	}
	return outn;

}

int picked_id( int b , int x , int y , Planetz*plz , Camera*cam )
{
	if( b != SDL_BUTTON_LEFT ) return -1;

	GLuint selectBuf[BUFSIZE];
	GLint hits;
	GLint viewport[4];

	glGetIntegerv (GL_VIEWPORT, viewport);

	glDepthRange(0.0, 1.0);

	glSelectBuffer (BUFSIZE, selectBuf);
	(void) glRenderMode (GL_SELECT);

	glInitNames();
	glPushName(0);

	glMatrixMode (GL_PROJECTION);
	glPushMatrix ();
	glLoadIdentity ();
	/*  create 5x5 pixel picking region near cursor location      */
	gluPickMatrix ((GLdouble) x, (GLdouble) (viewport[3] - y), 
			5.0, 5.0, viewport);
	gluPerspective(75.0, (double)gfx.width()/(double)gfx.height(), 1, 10000);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	cam->gl_lookat();

	plz->render();

	glMatrixMode (GL_PROJECTION);
	glPopMatrix ();
	glFlush ();

	hits = glRenderMode (GL_RENDER);
	plz->select( processHits (hits, selectBuf) );

	return 0;
}

#ifndef _WIN32
int main (int argc, char const* argv[])
#else 
int WINAPI WinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow )
#endif
{
	gfx.SDL_init(BASE_W,BASE_H);

	srand(static_cast<unsigned int>(timer.get_mms()));

	log_add(LOG_STREAM(stderr),LOG_PRINTER(vfprintf));

	FILE*f_log = fopen("planetz.log","w");
	log_add(LOG_STREAM(f_log),LOG_PRINTER(vfprintf));


#ifdef _RELEASE
	log_set_lev(INFO);
#endif

	gfx.GL_init();

	// FIXME: where should be this done?
	SigVideoResize.connect( 1 , boost::bind(&Gfx::CGfx::reshape_window,&gfx,_1,_2) );

	Camera cam( CAM_START_VECS );
//        SigKeyDown.connect( boost::bind(&Camera::on_key_down,&cam,_1) );
	SigMouseMotion.connect( boost::bind(&Camera::on_mouse_motion,&cam,_1,_2) );
	SigMouseButtonUp.connect( boost::bind(&Camera::on_button_up,&cam,_1,_2,_3) );
	SigMouseButtonDown.connect( 1 , boost::bind(&Camera::on_button_down,&cam,_1,_2,_3) );

	Planetz plz;
//        Planetz plz(&cam );
	
	Gfx::Background bkg( DATA("text.tga") , 0.8 );
	SigKeyDown.connect( boost::bind(&Gfx::Background::on_key_down,&bkg,_1) );
	SigMouseMotion.connect( boost::bind(&Gfx::Background::on_mouse_motion,&bkg,_1,_2) );
	SigMouseButtonUp.connect( boost::bind(&Gfx::Background::on_button_up,&bkg,_1,_2,_3) );
	SigMouseButtonDown.connect( 1 , boost::bind(&Gfx::Background::on_button_down,&bkg,_1,_2,_3) );


	SigMouseButtonDown.connect( 1 , boost::bind(picked_id,_1,_2,_3,&plz,&cam) );

	Saver saver( plz , cam );

#ifndef _NOGUI
	Gui gui;
	PlanetzLayout*pl = new PlanetzLayout(); 
	gui.set_layout(pl);

	pl->on_cam_speed_changed.connect( boost::bind(&Camera::set_speed,&cam,_1) );
	pl->on_pause_click.connect( boost::bind(pause_toggle) );
	pl->on_reset_click.connect( boost::bind(reset,&plz,&cam) );
	pl->on_save.connect( boost::bind(&Saver::save,&saver,_1) );
	pl->on_load.connect( boost::bind(&Saver::load,&saver,_1) );
	pl->on_load.connect( boost::bind(pause_anim) );

	pl->on_planet_add.connect( boost::bind(&Planetz::add,&plz,_1) );
	pl->on_planet_delete.connect( boost::bind(&Planetz::erase,&plz,_1) );
	plz.on_planet_select.connect( boost::bind(&PlanetzLayout::add_selected_planet,pl,_1) );
#endif

	bool running = true;

	log_printf(DBG,"Starting main loop\n");
	do
	{
		Timer::signal_all();
#ifndef _NOGUI
		gui.signal();
#endif
		cam.signal();

		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

		bkg.render();

		glDrawBuffer(GL_BACK);

		gfx.clear();

		cam.gl_lookat();
//                plz.lookat();

		if( !anim_pause )
			plz.update();
		plz.render();

#ifndef _NOGUI
		gui.render();
#endif

		SDL_GL_SwapBuffers();

		(running && (running &= event_handle() ));

		do_fps();
	}
	while( running );

	timer.stop();

	log_printf(INFO,"progs was running %lf seconds\n",timer.get());

	SDL_Quit();

	log_printf(DBG,"kthxbye\n");
	fclose(f_log);
	return 0;
}

