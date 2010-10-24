
#include <boost/signals.hpp>

#include "input.h"
#include "util/logger.h"
#include "gfx/gfx.h"

CInput::CInput()
{
	joy = new CMouseLD();

	if( joy->Init() == false ) {
		log_printf(CRITICAL,"Joy cannot be loaded\n");
		exit(0);
	}
}

CInput::~CInput()
{
	delete joy;
}

boost::signal<void (int,int,int)> SigKeyUp;
boost::signal<void (SDLKey,Uint16,Uint8)> SigKeyDown;
boost::signal<void (int,int)> SigMouseMotion;
boost::signal<void (int,int,int)> SigMouseButtonUp;
boost::signal<bool (int,int,int) , breaker > SigMouseButtonDown;
boost::signal<void (int,int)> SigVideoResize;

int event_handle()
{
	SDL_Event event;

	while( SDL_PollEvent(&event) )
	{
		switch( event.type )
		{
			case SDL_VIDEORESIZE:
				SigVideoResize( event.resize.w , event.resize.h );
				break;
			case SDL_KEYDOWN:
				SigKeyDown(
						event.key.keysym.sym
						,event.key.keysym.unicode
						,event.key.keysym.scancode);

				if( event.key.keysym.sym == SDLK_ESCAPE )
					return 0;
				break;
			case SDL_KEYUP:
				SigKeyUp(event.key.keysym.sym
						,event.key.keysym.unicode
						,event.key.keysym.scancode);
				break;
			case SDL_MOUSEBUTTONDOWN:
				SigMouseButtonDown(event.button.button,event.button.x,event.button.y);
				break;
			case SDL_MOUSEBUTTONUP:
				SigMouseButtonUp(event.button.button,event.button.x,event.button.y);
				break;
			case SDL_MOUSEMOTION:
				SigMouseMotion(event.motion.x,event.motion.y);
				break;
			case SDL_QUIT:
				return 0;
			default:
				break;
		}
	}
	return 1;
}

