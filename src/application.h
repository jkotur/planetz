#ifndef __APPLICATION_H__

#define __APPLICATION_H__

#include <cstdio>

#include "gfx/gfx.h"
#include "gfx/background.h"
#include "gfx/planetz_renderer.h"

#include "planetz_manager.h"

#include "constants.h"

#include "ui/ui.h"

#include "mem/memory_manager.h"
#include "mem/saver.h"

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

	Planetz planetz;
	Camera camera;
	UI ui;

	MEM::MemMgr memmgr;
	MEM::Saver saver;

	GFX::PlanetzRenderer plz;
	GFX::Background bkg;

	std::FILE*f_log;
};


#endif /* __APPLICATION_H__ */

