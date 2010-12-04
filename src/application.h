#ifndef __APPLICATION_H__

#define __APPLICATION_H__

#include <cstdio>

#include "phx/phx.h"
#include "gfx/gfx.h"
#include "gfx/background.h"
#include "gfx/planetz_renderer.h"
#include "gfx/arrow.h"

#include "constants.h"

#include "ui/ui.h"

#include "mem/data_flow_mgr.h"

#include "window.h"

#ifndef _RELEASE
#include "gfx/planetz_picker.h"
#include "debug/planet_printer.h"
#endif

class Application {
public:
	Application( Window& win );
	virtual ~Application();

	bool init();

#ifndef _RELEASE
	void test();
#endif

	void main_loop();

protected:
	void do_fps();

	void pause_toggle();
	void pause_anim();

	void reset();

	unsigned fps;
	float oldtime;

	bool anim_pause;

	Window& window;

	MEM::DataFlowMgr data_mgr;

	PHX::Phx phx;
	GFX::Gfx gfx;

	Camera camera;
	UI ui;
	PlanetzLayout*pl;

	GFX::PlanetzRenderer plz;
	GFX::Background bkg;

	std::FILE*f_log;

#ifndef _RELEASE
	GFX::Arrow * ox;
	GFX::Arrow * oy;
	GFX::Arrow * oz;

	GFX::PlanetzPicker picker;
	PlanetPrinter pprnt;
#endif
};


#endif /* __APPLICATION_H__ */

