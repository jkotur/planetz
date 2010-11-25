#include "application.h"

#include <cstdlib>

#include <string>

#include <boost/bind.hpp>

#ifndef _RELEASE
#include "gfx/arrow.h"
#endif

#include "util/vector.h"
#include "util/timer/timer.h"
#include "util/logger.h"

using boost::bind;

#define CAM_START_VECS Vector3(0,0,40),Vector3(0,0,0),Vector3(0,1,0)

Application::Application( Window& win )
	: fps(0) , anim_pause(true) ,
	  window( win )             ,
	  phx( data_mgr.getPhxMem() ),
	  camera( CAM_START_VECS )  ,
	  plz( data_mgr.getGfxMem() ) ,
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
	f_log = std::fopen(BIN("planetz.log").c_str(),"w");
	log_add(LOG_STREAM(f_log),LOG_PRINTER(std::vfprintf));
#ifdef _RELEASE
	log_set_lev(INFO);
#endif

	//
	// init graphics
	//
	if( !gfx.window_init(window.getW(),window.getH()) ) return false;

//        plz.setModels( memmgr.loadModels() ); // deprecated render mode

	//
	// init user interface
	//
	if( !ui.init() ) return false;

	// FIXME: where should be this done?
	ui.sigVideoResize.
		connect( 0 , bind(&Window::reshape_window,&window,_1,_2));
	ui.sigVideoResize.
		connect( 1 , bind(&GFX::Gfx::reshape_window,&gfx,_1,_2) );

	ui.sigMouseMotion.
		connect( bind(&Camera::on_mouse_motion,&camera,_1,_2) );
	ui.sigMouseButtonUp.
		connect( bind(&Camera::on_button_up,&camera,_1,_2,_3) );
	ui.sigMouseButtonDown.
		connect( 1 , bind(&Camera::on_button_down,&camera,_1,_2,_3));

	ui.sigKeyDown.
		connect( bind(&GFX::Background::on_key_down,&bkg,_1) );

	ui.sigMouseMotion.
		connect( bind(&GFX::Background::on_mouse_motion,&bkg,_1,_2));
	ui.sigMouseButtonUp.
		connect( bind(&GFX::Background::on_button_up,&bkg,_1,_2,_3));
	ui.sigMouseButtonDown.
		connect(1,bind(&GFX::Background::on_button_down,&bkg,_1,_2,_3));

#ifndef _NOGUI
	//
	// init graphical user interface
	//
	PlanetzLayout*pl = new PlanetzLayout(); 
	ui.gui.set_layout(pl);

	pl->on_cam_speed_changed.connect( bind(&Camera::set_speed,&camera,_1) );
	pl->on_pause_click.connect( bind(&Application::pause_toggle,this) );
	pl->on_reset_click.connect( bind(&Application::reset,this) );
	pl->on_save.connect( bind(&MEM::DataFlowMgr::save,&data_mgr,_1) );
	pl->on_load.connect( bind(&MEM::DataFlowMgr::load,&data_mgr,_1) );
	TODO( "handle save/load on DataFlowMgr level" );
	pl->on_load.connect( bind(&Application::pause_anim,this) );
	//pl->on_planet_add.connect( bind(&Planetz::add,&planetz,_1) );
	//pl->on_planet_delete.connect( bind(&Planetz::erase,&planetz,_1) );
	//planetz.on_planet_select.connect( bind(&PlanetzLayout::add_selected_planet,pl,_1) );
#endif

	gfx.add( &bkg    , 0 );
	gfx.add( &camera , 1 );
	gfx.add( &plz    , 2 );
#ifndef _NOGUI
	gfx.add( &ui     , 9 );
#endif
#ifndef _RELEASE
	ox = new GFX::Arrow(Vector3(1,0,0));
	oy = new GFX::Arrow(Vector3(0,1,0));
	oz = new GFX::Arrow(Vector3(0,0,1));
	gfx.add(  ox      );
	gfx.add(  oy      );
	gfx.add(  oz      );
#endif

	bkg.set_img(DATA("text.tga"));

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

		if( !anim_pause )
			phx.compute(3);
		gfx.render();

		(running && (running &= ui.event_handle() ));

		do_fps();
	}
	while( running );

	timer.stop();
}

Application::~Application()
{
#ifndef _RELEASE
	delete ox;
	delete oy;
	delete oz;
#endif

	log_printf(INFO,"Program is shutting down. It was running %lf seconds\n",timer.get());
	log_printf(DBG,"kthxbye\n");
	log_del(f_log);
	fclose(f_log);
}

void Application::do_fps()
{
	if( timer.get() - oldtime > 1 ) {
		oldtime = timer.get();
		log_printf(INFO,"fps: %d\n",fps);
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
	//planetz.clear();
	camera.set_perspective(CAM_START_VECS);
	anim_pause = true;
	//planetz.select(-1); // clear selection
}

#ifndef _RELEASE
#include "sphere/sphere.h"
#include "sphere/sphere_converter.h"

void Application::test()
{
}
#endif

