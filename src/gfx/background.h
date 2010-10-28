#ifndef __BACKGROUND_H__

#define __BACKGROUND_H__

#include <string>

#include "texture.h"
#include "drawable.h"

namespace GFX {

class Background : public Drawable {
public:
	Background( double , int , int );
	virtual ~Background();
	
	void set_img( const std::string&img );

	void on_reshape_window( int w , int h );
	void on_mouse_motion( int x , int y );
	void on_button_up( int , int , int );
	bool on_button_down( int , int , int );
	void on_key_down( int k );

	virtual void draw() const;
private:
	Texture*tex;

	double ox , oy;

	double size;

	int width , height;

	bool move;

	double sx , sy;
};

}

#endif /* __BACKGROUND_H__ */

