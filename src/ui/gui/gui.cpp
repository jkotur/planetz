#include <string>

#include "gfx/gfx.h"

#include "util/timer/timer.h"
#include "util/logger.h"
#include "constants.h"

#include "gui.h"
#include "layout.h"

using namespace CEGUI;

Gui::Gui()
	: layout(NULL)
{
}

Gui::~Gui()
{
	delete layout;
}

void Gui::set_layout( Layout*_l ) 
{
	if( layout ) delete layout;
	layout  = _l;
}

bool Gui::init()
{
	 try{ 
		init_throw();
	 } catch(CEGUI::InvalidRequestException e) {
	       log_printf(CRITICAL,"%s\n",e.getMessage().c_str());
	       return false;
	 }
	 return true;
}

void Gui::init_throw()
{
	SDL_ShowCursor(SDL_DISABLE);
	SDL_EnableUNICODE(1);
	SDL_EnableKeyRepeat(SDL_DEFAULT_REPEAT_DELAY, SDL_DEFAULT_REPEAT_INTERVAL);
	// FIXME: screen width & height in one place
	const SDL_VideoInfo* vidinfo = SDL_GetVideoInfo();
	int width = vidinfo->current_w;
	int height= vidinfo->current_h;

	renderer = new CEGUI::OpenGLRenderer(0,width,height);
	new System( renderer , 0 , 0 , 0 , "" , BIN("CEGUI.log") );

	DefaultResourceProvider* rp = static_cast<DefaultResourceProvider*>(
			CEGUI::System::getSingleton().getResourceProvider());

	rp->setResourceGroupDirectory("schemes",DATA("/gui/schemes/"));
	rp->setResourceGroupDirectory("imagesets", DATA("/gui/imagesets/"));
	rp->setResourceGroupDirectory("fonts", DATA("/gui/fonts/"));
	rp->setResourceGroupDirectory("layouts", DATA("/gui/layouts/"));
	rp->setResourceGroupDirectory("looknfeels", DATA("/gui/looknfeel/"));

	CEGUI::Imageset::setDefaultResourceGroup("imagesets");
	CEGUI::Font::setDefaultResourceGroup("fonts");
	CEGUI::Scheme::setDefaultResourceGroup("schemes");
	CEGUI::WidgetLookManager::setDefaultResourceGroup("looknfeels");
	CEGUI::WindowManager::setDefaultResourceGroup("layouts");

	CEGUI::SchemeManager::getSingleton().loadScheme( "QuadraticLook.scheme" );

	// load in a font.  The first font loaded automatically becomes the default font.
	if(! CEGUI::FontManager::getSingleton().isFontPresent( "Commonwealth-8" ) )
		CEGUI::FontManager::getSingleton().createFont( "Commonwealth.font" );

	System::getSingleton().setDefaultFont( "Commonwealth-8" );
	System::getSingleton().setDefaultMouseCursor( "QuadraticLook", "MouseArrow" );
	System::getSingleton().setDefaultTooltip( "QuadraticLook/Tooltip" );
}

void Gui::resize( int w , int h )
{
	renderer->setDisplaySize(CEGUI::Size(w, h));
	renderer->restoreTextures();
}

void Gui::render() const
{
	CEGUI::System::getSingleton().renderGUI();
}

void Gui::signal()
{
	CEGUI::System::getSingleton().injectTimePulse( timer.get_dt_mms() );
}

void Gui::on_mouse_motion( int x , int y )
{
	CEGUI::System::getSingleton().injectMousePosition(
			static_cast<float>(x),
			static_cast<float>(y)
			);
}

bool Gui::on_mouse_button_down( int b , int x, int y )
{
	switch ( b )
	{
		// handle real mouse buttons
		case SDL_BUTTON_LEFT:
			return CEGUI::System::getSingleton().injectMouseButtonDown(CEGUI::LeftButton);
		case SDL_BUTTON_MIDDLE:
			return CEGUI::System::getSingleton().injectMouseButtonDown(CEGUI::MiddleButton);
		case SDL_BUTTON_RIGHT:
			return CEGUI::System::getSingleton().injectMouseButtonDown(CEGUI::RightButton);

			// handle the mouse wheel
		case SDL_BUTTON_WHEELDOWN:
			return CEGUI::System::getSingleton().injectMouseWheelChange( -1 );
		case SDL_BUTTON_WHEELUP:
			return CEGUI::System::getSingleton().injectMouseWheelChange( +1 );
	}
	return false;
}

void Gui::on_mouse_button_up( int b , int x , int y )
{
	switch ( b )
	{
		case SDL_BUTTON_LEFT:
			CEGUI::System::getSingleton().injectMouseButtonUp(CEGUI::LeftButton);
			break;
		case SDL_BUTTON_MIDDLE:
			CEGUI::System::getSingleton().injectMouseButtonUp(CEGUI::MiddleButton);
			break;
		case SDL_BUTTON_RIGHT:
			CEGUI::System::getSingleton().injectMouseButtonUp(CEGUI::RightButton);
			break;
	}
}

void Gui::on_key_down( SDLKey k , Uint16 u , Uint8 code )
{
	CEGUI::System::getSingleton().injectKeyDown(code);

	if (u != 0) {
		CEGUI::System::getSingleton().injectChar(u);
	}
}

void Gui::on_key_up( int k )
{
	CEGUI::System::getSingleton().injectKeyUp(k);
}

