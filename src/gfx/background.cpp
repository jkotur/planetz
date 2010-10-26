#include <SDL/SDL_opengl.h>

#include "background.h"

#include "../constants.h"

#include "gfx.h"

using namespace Gfx;

Background::Background( double _s  , int w , int h )
	: size(_s) , width(w) , height(h) , move(false) ,
	  sx(0) , sy(0)
{
}

Background::~Background()
{
}

void Background::set_img( const std::string&img )
{
	tex = Texture::LoadTexture(img);
}

void Background::on_reshape_window( int w , int h )
{
	width = w;
	height= h;
}

void Background::on_key_down( int k )
{
	switch( k )
	{
	case LEFT_CAM_KEY_0:
	case LEFT_CAM_KEY_1:
		sx -= 0.1/(double)width;
		break;
	case RIGHT_CAM_KEY_0:
	case RIGHT_CAM_KEY_1:
		sx += 0.1/(double)width;
		break;
	}
}

void Background::on_mouse_motion( int x , int y )
{	
	if( !move ) return;

	int dx = x - ox;
	int dy = y - oy;
	ox = x;
	oy = y;

	/** sync with camera (almost works) */
	sx += (dx*CAM_ROT_SPEED)/PI2;
	sy -= (dy*CAM_ROT_SPEED)/PI2;

	/** following mose */
//        sx += (double)dx/(double)gfx.width(); 
//        sy -= (double)dy/(double)gfx.height();
}

bool Background::on_button_down( int b , int x , int y )
{
	if( b == ROT_BUTTON ) {
		move = true;
		ox = x;
		oy = y;
	}
	return false; // do not break signal
}

void Background::on_button_up( int b , int x , int y )
{
	if( b == ROT_BUTTON ) {
		move = false;
	}
}

void Background::render()
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);

//        glDepthFunc(GL_NEVER);
//        glDisable(GL_DEPTH_TEST);
//        glPolygonOffset
//        glDepthRange

	glDisable(GL_LIGHTING);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0,1,1,0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	if( tex ) {                      
		tex->bind();
		glEnable(GL_TEXTURE_2D);
	} else  glDisable(GL_TEXTURE_2D);

	glColor3f(1,1,1);
	glBegin(GL_QUADS);
	  glTexCoord2f(sx,sy+size);
	  glVertex2f(0,0);
	  glTexCoord2f(sx,sy);
	  glVertex2f(0,1);
	  glTexCoord2f(sx+size,sy);
	  glVertex2f(1,1);
	  glTexCoord2f(sx+size,sy+size);
	  glVertex2f(1,0);
	glEnd();

	glDisable(GL_TEXTURE_2D);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glPopAttrib();
}

