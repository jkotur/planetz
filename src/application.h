#ifndef __APPLICATION_H__

#define __APPLICATION_H__

#include <cstdio>

#include "gfx/gfx.h"
#include "gfx/background.h"

#include "planetz_manager.h"

#include "saver.h"

#include "constants.h"

#include "ui/ui.h"

class Application {
public:
	Application ();
	virtual ~Application();

	bool init();
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
	Saver saver;

	Gfx::CGfx gfx;
	Gfx::Background bkg;

	std::FILE*f_log;
};


#endif /* __APPLICATION_H__ */

