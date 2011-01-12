#include "ui.h"

#include <boost/bind.hpp>

#include "util/logger.h"

using boost::bind;

UI::UI::UI()
{
}

UI::UI::~UI()
{
	if( joy ) delete joy;
}

bool UI::UI::init()
{
	log_printf(INFO,"Starting UI\n");

	//
	// Setup mouse
	//
	joy = new CMouseLD();

	if( joy->Init() == false ) {
		log_printf(CRITICAL,"Joy cannot be loaded\n");
		return false;
	}

	//
	// Setup gui
	// 
	if( !gui.init() ) return false;

	sigKeyUp.connect( boost::bind(&Gui::on_key_up,gui,_1) );
	sigKeyDown.connect( boost::bind(&Gui::on_key_down,gui,_1,_2,_3) );
	sigMouseMotion.connect( boost::bind(&Gui::on_mouse_motion,gui,_1,_2) );
	sigMouseButtonUp.connect( boost::bind(&Gui::on_mouse_button_up,gui,_1,_2,_3) );
	sigMouseButtonDown.connect( 0 , boost::bind(&Gui::on_mouse_button_down,gui,_1,_2,_3) );

	sigVideoResize.connect( 0 , boost::bind(&CEGUI::OpenGLRenderer::grabTextures,gui.renderer) );
	sigVideoResize.connect( 3 , boost::bind(&Gui::resize,gui,_1,_2) );

	return true;
}

void UI::UI::draw() const 
{
	gui.render();
}

void UI::UI::signal()
{
	gui.signal();
}

int UI::UI::event_handle()
{
	SDL_Event event;

	while( SDL_PollEvent(&event) )
	{
		switch( event.type )
		{
			case SDL_VIDEORESIZE:
				sigVideoResize( event.resize.w , event.resize.h );
				break;
			case SDL_KEYDOWN:
				sigKeyDown(
						event.key.keysym.sym
						,event.key.keysym.unicode
						,event.key.keysym.scancode);

				if( event.key.keysym.sym == SDLK_ESCAPE )
					return 0;
				break;
			case SDL_KEYUP:
				sigKeyUp(event.key.keysym.sym
						,event.key.keysym.unicode
						,event.key.keysym.scancode);
				break;
			case SDL_MOUSEBUTTONDOWN:
				sigMouseButtonDown(event.button.button,event.button.x,event.button.y);
				break;
			case SDL_MOUSEBUTTONUP:
				sigMouseButtonUp(event.button.button,event.button.x,event.button.y);
				break;
			case SDL_MOUSEMOTION:
				sigMouseMotion(event.motion.x,event.motion.y);
				break;
			case SDL_QUIT:
				return 0;
			default:
				break;
		}
	}
	return 1;
}

void UI::UI::add_listener( InputListener* lst , int level )
{
	sigKeyUp          .   connect( level
			, bind(&InputListener::on_key_up      ,lst,_1,_2,_3));
	sigKeyDown        .   connect( level
			, bind(&InputListener::on_key_down    ,lst,_1,_2,_3));
	sigMouseMotion    .   connect( level
			, bind(&InputListener::on_mouse_motion,lst,_1,_2   ));
	sigMouseButtonUp  .   connect( level
			, bind(&InputListener::on_button_up   ,lst,_1,_2,_3));
	sigMouseButtonDown.   connect( level
			, bind(&InputListener::on_button_down ,lst,_1,_2,_3));
}

void UI::UI::del_listener( InputListener* lst )
{
	sigKeyUp          .disconnect(bind(&InputListener::on_key_up      ,lst,_1,_2,_3));
	sigKeyDown        .disconnect(bind(&InputListener::on_key_down    ,lst,_1,_2,_3));
	sigMouseMotion    .disconnect(bind(&InputListener::on_mouse_motion,lst,_1,_2   ));
	sigMouseButtonUp  .disconnect(bind(&InputListener::on_button_up   ,lst,_1,_2,_3));
	sigMouseButtonDown.disconnect(bind(&InputListener::on_button_down ,lst,_1,_2,_3));
}

