#include "application.h"

#include <cstdlib>

#include <SDL/SDL.h>
#include <SDL/SDL_opengl.h>

#include <boost/bind.hpp>

#include "util/vector.h"
#include "util/timer/timer.h"
#include "util/logger.h"


using boost::bind;

#define CAM_START_VECS Vector3(0,0,10),Vector3(0,0,0),Vector3(0,1,0)

Application::Application()
	: fps(0) , anim_pause(true) ,
	  camera( CAM_START_VECS )  ,
	  saver( planetz , camera )        ,
	  bkg( 0.8 , BASE_W , BASE_H )
{
}

bool Application::init()
{
	//
	// seed da rand
	//
	std::srand(static_cast<unsigned int>(timer.get_mms()));

	//
	// setup new old time 
	//
	oldtime = timer.get_dt_s();

	//
	// init logger
	//
	log_add(LOG_STREAM(stderr),LOG_PRINTER(std::vfprintf));
	f_log = std::fopen("planetz.log","w");
	log_add(LOG_STREAM(f_log),LOG_PRINTER(std::vfprintf));
#ifdef _RELEASE
	log_set_lev(INFO);
#endif

	//
	// init graphics
	//
	if( !gfx.SDL_init(BASE_W,BASE_H) ) return false;
	if( !gfx.GL_init()               ) return false;

	bkg.set_img(DATA("text.tga"));

	//
	// init user interface
	//
	if( !ui.init()                   ) return false;

	// FIXME: where should be this done?
	ui.sigVideoResize.
		connect( 1 , bind(&Gfx::CGfx::reshape_window,&gfx,_1,_2) );
	ui.sigVideoResize.
		connect( 1 , bind(&Gfx::Background::on_reshape_window,&bkg,_1,_2));

	ui.sigMouseMotion.
		connect( bind(&Camera::on_mouse_motion,&camera,_1,_2) );
	ui.sigMouseButtonUp.
		connect( bind(&Camera::on_button_up,&camera,_1,_2,_3) );
	ui.sigMouseButtonDown.
		connect( 1 , bind(&Camera::on_button_down,&camera,_1,_2,_3));

	ui.sigKeyDown.
		connect( bind(&Gfx::Background::on_key_down,&bkg,_1) );

	ui.sigMouseMotion.
		connect( bind(&Gfx::Background::on_mouse_motion,&bkg,_1,_2));
	ui.sigMouseButtonUp.
		connect( bind(&Gfx::Background::on_button_up,&bkg,_1,_2,_3));
	ui.sigMouseButtonDown.
		connect(1,bind(&Gfx::Background::on_button_down,&bkg,_1,_2,_3));

#ifndef _NOGUI
	//
	// init graphical user interface
	//
	PlanetzLayout*pl = new PlanetzLayout(); 
	ui.gui.set_layout(pl);

	pl->on_cam_speed_changed.connect( bind(&Camera::set_speed,&camera,_1) );
	pl->on_pause_click.connect( bind(&Application::pause_toggle,this) );
	pl->on_reset_click.connect( bind(&Application::reset,this) );
	pl->on_save.connect( bind(&Saver::save,&saver,_1) );
	pl->on_load.connect( bind(&Saver::load,&saver,_1) );
	pl->on_load.connect( bind(&Application::pause_anim,this) );
	pl->on_planet_add.connect( bind(&Planetz::add,&planetz,_1) );
	pl->on_planet_delete.connect( bind(&Planetz::erase,&planetz,_1) );
	planetz.on_planet_select.connect( bind(&PlanetzLayout::add_selected_planet,pl,_1) );
#endif

	return true;
}

void Application::main_loop()
{
	bool running = true;

	log_printf(DBG,"Starting main loop\n");
	do
	{
		Timer::signal_all();
#ifndef _NOGUI
		ui.signal();
#endif
		camera.signal();

		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

		bkg.render();

		glDrawBuffer(GL_BACK);

		gfx.clear();

		camera.gl_lookat();

		if( !anim_pause )
			planetz.update();
		planetz.render();

#ifndef _NOGUI
		ui.render();
#endif

		SDL_GL_SwapBuffers();

		(running && (running &= ui.event_handle() ));

		do_fps();
	}
	while( running );

	timer.stop();
}

Application::~Application()
{
	log_printf(INFO,"Program is shutting down. It was running %lf seconds\n",timer.get());

	log_printf(DBG,"kthxbye\n");
	fclose(f_log);
}

void Application::do_fps()
{
	if( timer.get() - oldtime > 1 ) {
		oldtime = timer.get();
		log_printf(INFO,"fps: %d\n",fps);
//                Gfx::Hud::fps = fps;
		fps = 0;
	}
	fps++;
}

void Application::pause_toggle()
{
	anim_pause = !anim_pause;
}

void Application::pause_anim()
{
	anim_pause = true;
}

void Application::reset() // Planetz*pl , Camera*c )
{
	planetz.clear();
	camera.set_perspective(CAM_START_VECS);
	anim_pause = true;
	planetz.select(-1); // clear selection
}

