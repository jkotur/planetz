#ifndef __GUI_H__

#define __GUI_H__

#include <GL/glew.h>
  
#include <CEGUI.h>
#include <CEGUIDefaultResourceProvider.h>
#include <RendererModules/OpenGLGUIRenderer/openglrenderer.h>

#include "layout.h"
#include "planetzlayout.h"

class Gui {
public:
	Gui ();
	virtual ~Gui();

	bool init();
	void init_throw();
	void resize( int x, int y );
	void render();
	void signal();

	void on_mouse_motion( int , int );
	bool on_mouse_button_down( int , int , int );
	void on_mouse_button_up( int , int , int );
	void on_key_down( SDLKey k , Uint16 u , Uint8 code );
	void on_key_up( int );

	void set_layout( Layout*l );

	// FIXME: public?
	CEGUI::OpenGLRenderer*renderer;
private:
	bool add_item( const CEGUI::EventArgs& e );

	Layout*layout;

	boost::signals::connection cons[7];
	
};



#endif /* __GUI_H__ */

