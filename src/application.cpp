#include "application.h"

#include <cstdlib>

#include <string>

#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>

#include "util/vector.h"
#include "util/timer/timer.h"
#include "util/logger.h"

using boost::bind;

#define CAM_START_VECS Vector3(0,0,4),Vector3(0,0,0),Vector3(0,1,0)

Application::Application( Window& win , Config& cfg )
	: fps(0)
	, anim_pause(true)
	, phx_frames(1)
	, window( win )
	, config( cfg )
	, phx( data_mgr.getPhxMem() )
	, picker( data_mgr.getGfxMem(), 5, 5 , win.getW() , win.getH() )
	, camera( CAM_START_VECS )
	, plz( data_mgr.getGfxMem() )
	, trace( *data_mgr.getGfxMem() , cfg )
	, bkg( 0.8 , win.getW() , win.getH() )
	, phcleaner( data_mgr.getPhxMem() )
#ifndef _RELEASE
	, pprnt( data_mgr.getPhxMem() )
#endif
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
		connect( 2 , bind(&UI::PlanetzPicker::resize,&picker,_1,_2) );

	ui.add_listener( &setter , 1 );
	ui.add_listener( &camera , 2 );

	ui.sigKeyDown.
		connect( bind(&GFX::Background::on_key_down,&bkg,_1) );

	ui.sigMouseMotion.
		connect( bind(&GFX::Background::on_mouse_motion,&bkg,_1,_2));
	ui.sigMouseButtonUp.
		connect( bind(&GFX::Background::on_button_up,&bkg,_1,_2,_3));
	ui.sigMouseButtonDown.
		connect(1,bind(&GFX::Background::on_button_down,&bkg,_1,_2,_3));

	ui.sigMouseButtonDown.
		connect( bind(&UI::PlanetzPicker::on_button_down,&picker,_1,_2,_3) );

	camera.sigCamChanged.
		connect( 2 , bind(&GFX::DeferRender::on_camera_angle_changed,&plz,_1) );

	picker.sigPlanetPicked.
		connect( bind(&Application::set_picked_planet,this,_1) );
#ifndef _RELEASE
	picker.sigPlanetPicked.
		connect( bind(&PlanetPrinter::print,&pprnt,_1));
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
	ui.set_layout(pl);


//        pl->on_cam_speed_changed.connect(
//                        bind(	&UI::CameraMgr::update,&camera
//                                ,UI::CameraMgr::FREELOOK,&boost::lambda::_1) );
	pl->on_cam_speed_changed.connect( bind(&Application::set_cam_speed,this,_1) );
	pl->on_sim_speed_changed.connect( bind(&Application::set_phx_speed,this,_1) );
	pl->on_pause_click.connect( bind(&Application::pause_toggle,this) );
	pl->on_reset_click.connect( bind(&Application::reset,this) );
	pl->on_save.connect( 1, bind(&MEM::DataFlowMgr::save,&data_mgr,_1) );
	pl->on_save.connect( 0, bind(&MEM::MISC::PlanetHolderCleaner::forceFilter,&phcleaner) );
	pl->on_load.connect( bind(&MEM::DataFlowMgr::load,&data_mgr,_1) );
	pl->on_load.connect( bind(&Application::pause_anim,this) );
	pl->on_load.connect( bind(&GFX::PlanetsTracer::clear,&trace) );
	pl->on_load.connect( bind(&UI::CameraMgr::clear,&camera) );
	pl->on_load.connect( bind(&UI::PlanetzSetter::clear,&setter) );
	pl->on_load.connect( bind(&PlanetzLayout::hide_show_window,pl) );
	pl->on_config_changed.connect(bind(&GFX::Gfx::update_configuration,&gfx,_1));
	pl->on_config_changed.connect(bind(&Application::update_configuration,this,_1));
	pl->on_planet_changed.connect( bind(&UI::PlanetzSetter::update,&setter,_1) );
	pl->on_planet_change.connect( bind(&UI::PlanetzSetter::change,&setter,_1) );
	pl->on_planet_add.connect( bind(&MEM::DataFlowMgr::createPlanet,&data_mgr,_1) );
	pl->on_planet_add.connect( bind(&UI::PlanetzSetter::clear,&setter) );
	pl->on_planet_delete.connect( bind(&MEM::DataFlowMgr::removePlanet,&data_mgr,_1) );
	pl->on_planet_delete.connect( bind(&MEM::MISC::PlanetHolderCleaner::notifyCheckNeeded,&phcleaner) );
	//planetz.on_planet_select.connect( bind(&PlanetzLayout::add_selected_planet,pl,_1) );

	setter.on_planet_changed.connect( bind(&PlanetzLayout::update_add_win,pl,_1) );
#endif

//        data_mgr.load(DATA("saves/qsave.sav"));

//        gfx.add( &bkg    , 0 );
	gfx.add( &camera , 1 );
	gfx.add( &plz    , 2 );
	gfx.add( &trace  , 3 );
	gfx.add( &setter , 4 );
#ifndef _NOGUI
	gfx.add( &ui     , 9 );
#endif

	update_configuration( config );
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
			//log_printf(INFO,"phx.compute(%u) running time: %.2fms\n", phx_frames, t.get_ms());
		}
		phcleaner.work();
//                pt.refresh();
		gfx.render();

		(running && (running &= ui.event_handle() ));

		do_fps();
	}
	while( running );

	timer.stop();
}

Application::~Application()
{
	pp = MEM::MISC::PhxPlanet();
	log_printf(INFO,"Program is shutting down. It was running %lf seconds\n",timer.get());
	log_printf(DBG,"kthxbye\n");
	log_del(f_log);
	fclose(f_log);
}

void Application::set_picked_planet( int id )
{
	if( id < 0 ) {
		pp = MEM::MISC::PhxPlanet();
		camera.request(UI::CameraMgr::ZOOMIN);
		pl->hide_show_window();
	}
	else {
		pp = data_mgr.getPhxMem()->getPlanet( id );
		camera.request(UI::CameraMgr::ZOOMIN,&pp);
		pl->show_show_window( pp );
	}
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

void Application::reset()
{
	pl->hide_show_window();

	camera.set_perspective(CAM_START_VECS);
	camera.clear();

	anim_pause = true;
	trace.stop();

	data_mgr.dropPlanets();
}

#ifndef _RELEASE
void Application::test()
{
}
#endif

void Application::update_configuration( const Config &cfg )
{
	phx.enableClusters( cfg.get<bool>( "phx.clusters" ) );

	MEM::MISC::PlanetHolderCleaner::FilteringPolicy policy;
	policy = cfg.get<bool>( "trace.enable" ) ?
		MEM::MISC::PlanetHolderCleaner::Always :
		MEM::MISC::PlanetHolderCleaner::Rarely;
	phcleaner.setFilteringPolicy( policy );
}
