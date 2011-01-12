#include "camera.h"

#include <GL/glew.h>

#include "./util/logger.h"
#include "./util/timer/timer.h"

#include "constants.h"

using UI::CamFreeLook;
using UI::CamLocked;
using UI::CamZoomIn;

CamFreeLook::CamFreeLook()
	: ox(0) , oy(0) , move_speed(0.0)
	, rot(false) 
{
}

CamFreeLook::~CamFreeLook()
{
}

void CamFreeLook::born( Matrix state , void*data )
{
	matrix = state;
	learn(data);
}

void CamFreeLook::learn( void * data )
{
	move_speed = *(double*)data;
}

UI::Camera::Matrix CamFreeLook::work()
{
	signal();
	return matrix;
}

UI::Camera::Matrix CamFreeLook::die ()
{
	return matrix; // always ready to die
}

void CamFreeLook::on_mouse_motion( int x , int y )
{
	if( !rot ) return;
	int dx = x - ox;
	int dy = y - oy;
	ox = x;
	oy = y;

	double angle_x = dx*CAM_ROT_SPEED;
	double angle_y = dy*CAM_ROT_SPEED;

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glRotatef( angle_x , 0, 1, 0 );
	glRotatef( angle_y , 1, 0, 0 );
	glMultMatrixf( matrix );
	glGetFloatv(GL_MODELVIEW_MATRIX,matrix);
	glPopMatrix();
}

void CamFreeLook::on_button_up( int b , int , int )
{
	if( b == ROT_BUTTON )
		rot = false;
}

bool CamFreeLook::on_button_down( int b , int x , int y )
{
	if( b == ROT_BUTTON ) {
		rot = true;
		ox = x;
		oy = y;
	}
	return false; // do not end mouse singal
}

void CamFreeLook::signal()
{
	Uint8 *keystate = SDL_GetKeyState(NULL);
	float ds = (move_speed*timer.get_dt_s()+BASE_CAM_SPEED);

	if( keystate[FWD_CAM_KEY_0  ] || keystate[FWD_CAM_KEY_1  ] )
		matrix[14] += ds;
	if( keystate[BCK_CAM_KEY_0  ] || keystate[BCK_CAM_KEY_1  ] )
		matrix[14] -= ds;
	if( keystate[LEFT_CAM_KEY_0 ] || keystate[LEFT_CAM_KEY_1 ] )
		matrix[12] += ds;
	if( keystate[RIGHT_CAM_KEY_0] || keystate[RIGHT_CAM_KEY_1] )
		matrix[12] -= ds;
	if( keystate[UP_CAM_KEY_0   ] || keystate[UP_CAM_KEY_1   ] )
		matrix[13] -= ds;
	if( keystate[DOWN_CAM_KEY_0 ] || keystate[DOWN_CAM_KEY_1 ] )
		matrix[13] += ds;
}

CamLocked::CamLocked()
{
}

CamLocked::~CamLocked()
{
}

void CamLocked::born( Matrix state , void*data )
{
}

void CamLocked::learn( void * data )
{
}

UI::Camera::Matrix CamLocked::work()
{
	return NULL;
}

UI::Camera::Matrix CamLocked::die ()
{
	return NULL;
}

CamZoomIn::CamZoomIn()
{
}

CamZoomIn::~CamZoomIn()
{
}

void CamZoomIn::born( Matrix state , void*data )
{
}

void CamZoomIn::learn( void * data )
{
}

UI::Camera::Matrix CamZoomIn::work()
{
	return NULL;
}

UI::Camera::Matrix CamZoomIn::die ()
{
	return NULL;
}

