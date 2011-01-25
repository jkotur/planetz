/** 
 * @file application.h
 * @brief 
 * @author Jakub Kotur
 * @version 1.0
 * @date 2010-12-30
 */

#ifndef __APPLICATION_H__

#define __APPLICATION_H__

#include <cstdio>

#include "phx/phx.h"
#include "gfx/gfx.h"
#include "gfx/background.h"
#include "gfx/planetz_tracer.h"
#include "gfx/deffered_renderer.h"

#include "constants.h"

#include "ui/ui.h"
#include "ui/planetz_setter.h"
#include "ui/planetz_picker.h"
#include "ui/camera_manager.h"

#include "mem/data_flow_mgr.h"

#include "window.h"

#include "util/config.h"
#include "mem/misc/holder_cleaner.h"

#ifndef _RELEASE
#include "debug/planet_printer.h"
#endif

/**
 * Główna klasa programu.Zawiera wszystkie główne moduły,
 * oraz odpowiedzialna jest za kontrolę głównej pętli programu
 */
class Application {
public:
	/** 
	 * @brief Jedyny konstruktor aplikacji.
	 * 
	 * @param win okno na którym ma działać aplikacja
	 * @param cfg konfiguracja programu
	 */
	Application( Window& win , Config& cfg );
	virtual ~Application();

	/** 
	 * @brief wykonuje wszystkie niezbędne czynności,
	 * aby zainicjalizować aplikację.
	 * 
	 * @return true jeśli się powiedzie, false wpp
	 */
	bool init();

#ifndef _RELEASE
	void test();
#endif

	/** 
	 * @brief główna pętla aplikacji. W niej liczone są kolejne klatki fizyki
	 * i wyświetlane planety na ekranie.
	 */
	void main_loop();

	void set_phx_speed( double s )
	{	phx_frames = (unsigned)s; }

	void set_cam_speed( double s )
	{	camera.update(UI::CameraMgr::FREELOOK,(void*)&s); }
protected:
	void set_picked_planet( int id );

	void do_fps();

	void pause_toggle();
	void pause_anim();

	void reset();

	void update_configuration( const Config &cfg );

	MEM::MISC::PhxPlanet pp;

	unsigned fps;
	float oldtime;

	bool anim_pause;

	unsigned phx_frames;

	Window& window;
	Config& config;

	MEM::DataFlowMgr data_mgr;

	PHX::Phx phx;
	GFX::Gfx gfx;

	UI::PlanetzSetter setter;
	UI::PlanetzPicker picker;
	UI::CameraMgr camera;
	UI::UI ui;
	PlanetzLayout*pl;

	GFX::DeferRender plz;
	GFX::PlanetsTracer trace;
	GFX::Background bkg;

	std::FILE*f_log;

	MEM::MISC::PlanetHolderCleaner phcleaner;

#ifndef _RELEASE
	PlanetPrinter pprnt;
#endif
};


#endif /* __APPLICATION_H__ */

