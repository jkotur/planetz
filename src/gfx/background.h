#ifndef __BACKGROUND_H__

#define __BACKGROUND_H__

#include <string>

#include "texture.h"

namespace Gfx {

class Background {
public:
	Background( const std::string& img , double );
	virtual ~Background();
	
	void on_mouse_motion( int x , int y );
	void on_button_up( int , int , int );
	bool on_button_down( int , int , int );
	void on_key_down( int k );

	void render();
private:
	Texture*tex;

	double ox , oy;

	double size;

	bool move;

	double sx , sy;
};

}

#endif /* __BACKGROUND_H__ */

