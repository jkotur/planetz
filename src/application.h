#ifndef __APPLICATION_H__

#define __APPLICATION_H__

#include <cstdio>

#include "gfx/gfx.h"
#include "gfx/background.h"
#include "gfx/planetz_renderer.h"
#include "gfx/arrow.h"

#include "constants.h"

#include "ui/ui.h"

#include "mem/data_flow_mgr.h"

#include "window.h"

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

	GFX::Gfx gfx;

	Camera camera;
	UI ui;

	MEM::DataFlowMgr data_mgr;

	GFX::PlanetzRenderer plz;
	GFX::Background bkg;

	std::FILE*f_log;

#ifndef _RELEASE
	GFX::Arrow * ox;
	GFX::Arrow * oy;
	GFX::Arrow * oz;
#endif
};


#endif /* __APPLICATION_H__ */

