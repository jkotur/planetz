#ifndef __APPLICATION_H__

#define __APPLICATION_H__

#include <cstdio>

#include "gfx/gfx.h"
#include "gfx/background.h"

#include "planetz_manager.h"

#include "constants.h"

#include "ui/ui.h"

#include "mem/memory_manager.h"
#include "mem/saver.h"


class Application {
public:
	Application ();
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

	Planetz planetz;
	Camera camera;
	UI ui;
	MEM::Saver saver;

	GFX::Gfx gfx;
	GFX::Background bkg;

	MEM::MemMgr memmgr;

	std::FILE*f_log;
};


#endif /* __APPLICATION_H__ */

