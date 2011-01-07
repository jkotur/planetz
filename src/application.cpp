#include "application.h"

#include <cstdlib>

#include <string>

#include <boost/bind.hpp>

#include "util/vector.h"
#include "util/timer/timer.h"
#include "util/logger.h"

using boost::bind;

#define CAM_START_VECS Vector3(0,0,40),Vector3(0,0,0),Vector3(0,1,0)

Application::Application( Window& win , Config& cfg )
	: fps(0)
	, anim_pause(true)
	, phx_frames(1)
	, window( win )
	, config( cfg )
	, phx( data_mgr.getPhxMem() )
	, camera( CAM_START_VECS )
	, plz( data_mgr.getGfxMem() )
	, trace( *data_mgr.getGfxMem() , cfg )
	, bkg( 0.8 , win.getW() , win.getH() )
	, phcleaner( data_mgr.getPhxMem() )
	, picker( data_mgr.getGfxMem(), 3, 3 , win.getW() , win.getH() )
	, pprnt( data_mgr.getPhxMem(), &picker )
	, pt( data_mgr.getPhxMem(), &picker, &camera )
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

	plz.setMaterials( data_mgr.loadMaterials() );
	plz.setTextures ( data_mgr.loadTextures () );

	//
	// init user interface
	//
	if( !ui.init() ) return false;

	// FIXME: where should be this done?
	ui.sigVideoResize.
		connect( 0 , bind(&Window::reshape_window,&window,_1,_2));
	ui.sigVideoResize.
		connect( 1 , bind(&GFX::Gfx::reshape_window,&gfx,_1,_2) );
	ui.sigVideoResize.
		connect( 2 , bind(&GFX::PlanetzPicker::resize,&picker,_1,_2) );

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

	camera.sigAngleChanged.
		connect( bind(&GFX::DeferRender::on_camera_angle_changed,&plz,_1) );

#ifndef _RELEASE
	ui.sigMouseButtonDown.
		connect(1,bind(&PlanetPrinter::on_button_down,&pprnt,_1,_2,_3));
	ui.sigMouseButtonDown.
		connect(1,bind(&PlanetTracer::on_button_down,&pt,_1,_2,_3));
#endif

#ifndef _NOGUI
	//
	// init graphical user interface
	//
	try {
		pl = new PlanetzLayout( config ); 
	} catch(CEGUI::InvalidRequestException e) {
		log_printf(CRITICAL,"CEGUI exception: %s\n",e.getMessage().c_str());
		return false;
	}
	ui.gui.set_layout(pl);

	pl->on_cam_speed_changed.connect( bind(&Camera::set_speed,&camera,_1) );
	pl->on_sim_speed_changed.connect( bind(&Application::set_phx_speed,this,_1) );
	pl->on_pause_click.connect( bind(&Application::pause_toggle,this) );
	pl->on_reset_click.connect( bind(&Application::reset,this) );
	pl->on_save.connect( bind(&MEM::DataFlowMgr::save,&data_mgr,_1) );
	pl->on_load.connect( bind(&MEM::DataFlowMgr::load,&data_mgr,_1) );
	pl->on_load.connect( bind(&Application::pause_anim,this) );
	pl->on_load.connect( bind(&GFX::PlanetsTracer::clear,&trace) );
	pl->on_config_changed.connect(bind(&GFX::Gfx::update_configuration,&gfx,_1));
	//pl->on_planet_add.connect( bind(&Planetz::add,&planetz,_1) );
	//pl->on_planet_delete.connect( bind(&Planetz::erase,&planetz,_1) );
	//planetz.on_planet_select.connect( bind(&PlanetzLayout::add_selected_planet,pl,_1) );
#endif

//        data_mgr.load(DATA("saves/qsave.sav"));

//        gfx.add( &bkg    , 0 );
	gfx.add( &camera , 1 );
	gfx.add( &plz    , 2 );
	gfx.add( &trace  , 3 );
#ifndef _NOGUI
	gfx.add( &ui     , 9 );
#endif

	gfx.update_configuration( config );

	camera.init();

//        bkg.set_img(DATA("text.tga"));

	data_mgr.registerCam( &camera );

	phx.registerCleaner( &phcleaner );

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
		{
			//Timer t;
			phx.compute(phx_frames);
			//log_printf(DBG, "phx.compute(%u) running time: %.2fms\n", phx_frames, timer.get_dt_ms());
		}
		pt.refresh();
		gfx.render();

		(running && (running &= ui.event_handle() ));

		do_fps();
		phcleaner.work();
	}
	while( running );

	timer.stop();
}

Application::~Application()
{
	log_printf(INFO,"Program is shutting down. It was running %lf seconds\n",timer.get());
	log_printf(DBG,"kthxbye\n");
	log_del(f_log);
	fclose(f_log);
}

void Application::do_fps()
{
	if( timer.get() - oldtime > 1 ) {
		oldtime = timer.get();
		pl->update_fps(fps);
//                log_printf(INFO,"fps: %d\n",fps);
		fps = 0;
	}
	fps++;
}

void Application::pause_toggle()
{
	anim_pause = !anim_pause;

	if( anim_pause )
		trace.stop();
	else	trace.start();
}

void Application::pause_anim()
{
	anim_pause = true;
	trace.stop();
}

void Application::reset() // Planetz*pl , Camera*c )
{
	//planetz.clear();
	camera.set_perspective(CAM_START_VECS);
	anim_pause = true;
	trace.stop();
	//planetz.select(-1); // clear selection
}

#ifndef _RELEASE
void Application::test()
{
}
#endif

