#include "camera.h"

#include <GL/glew.h>

#include "./util/logger.h"
#include "./util/timer/timer.h"

#include "constants.h"

Camera::Camera( const Vector3& _p ,
		const Vector3& _l ,
		const Vector3& _u )
	: pos(_p) , lookat(_l) , up(_u)
	, ox(0) , oy(0) , speed(0.2) , move_speed(0.2)
	, rot(false)
{
	init();
}

Camera::~Camera()
{
	log_printf(DBG,"[DEL] Camera is dying\n");
}

void Camera::init()
{
	right = up;
	right.cross( pos - lookat );
	right.normalize();

	forward = lookat - pos;
	forward.normalize();
}

void Camera::on_key_down( int k )
{
	switch( k )
	{
	case SDLK_UP:
		pos += forward*move_speed;
		lookat += forward*move_speed;
		break;
	case SDLK_DOWN:
		pos -= forward*move_speed;
		lookat -= forward*move_speed;
		break;
	case LEFT_CAM_KEY_0:
		pos -= right*move_speed;
		lookat -= right*move_speed;
		break;
	case RIGHT_CAM_KEY_0:
		pos += right*move_speed;
		lookat += right*move_speed;
		break;
	}
}

void Camera::on_mouse_motion( int x , int y )
{	// FIXME: to potrzebuje optymalizacji
	if( !rot ) return;
	int dx = x - ox;
	int dy = y - oy;
	ox = x;
	oy = y;

	double angle_x = dx*CAM_ROT_SPEED;
	double angle_y = dy*CAM_ROT_SPEED;

	lookat-=pos;
	lookat.rotate( up , angle_x );
	right.rotate( up , angle_x );
	forward.rotate( up , angle_x );

	up.rotate( right , angle_y );
	lookat.rotate( right ,  angle_y );
	forward.rotate( right , angle_y );
	lookat+=pos;

}

bool Camera::on_button_down( int b , int x , int y )
{
	if( b == ROT_BUTTON ) {
		rot = true;
		ox = x;
		oy = y;
	}
	return false; // do not end mouse singal
}

void Camera::on_button_up( int b , int x , int y )
{
	if( b == ROT_BUTTON ) {
		rot = false;
	}
}

void Camera::draw() const 
{
	gluLookAt(
		pos.x , pos.y , pos.z ,
		lookat.x , lookat.y , lookat.z ,
		up.x , up.y , up.z
	);
}

void Camera::signal()
{
	Uint8 *keystate = SDL_GetKeyState(NULL);

	if ( keystate[FWD_CAM_KEY_0] || keystate[FWD_CAM_KEY_1] ) {
		pos +=    forward*(move_speed*timer.get_dt_s()+BASE_CAM_SPEED);
		lookat += forward*(move_speed*timer.get_dt_s()+BASE_CAM_SPEED);
	}
	if ( keystate[BCK_CAM_KEY_0]  || keystate[BCK_CAM_KEY_1] ) {
		pos -=    forward*(move_speed*timer.get_dt_s()+BASE_CAM_SPEED);
		lookat -= forward*(move_speed*timer.get_dt_s()+BASE_CAM_SPEED);
	}
	if ( keystate[LEFT_CAM_KEY_0] || keystate[LEFT_CAM_KEY_1] ) {
		pos -=    right*(move_speed*timer.get_dt_s()+BASE_CAM_SPEED);
		lookat -= right*(move_speed*timer.get_dt_s()+BASE_CAM_SPEED);
	}
	if ( keystate[RIGHT_CAM_KEY_0] || keystate[RIGHT_CAM_KEY_1] ) {
		pos +=    right*(move_speed*timer.get_dt_s()+BASE_CAM_SPEED);
		lookat += right*(move_speed*timer.get_dt_s()+BASE_CAM_SPEED);
	}
	if ( keystate[UP_CAM_KEY_0] || keystate[UP_CAM_KEY_1] ) {
		Vector3 tmp = up*(move_speed*timer.get_dt_s()+BASE_CAM_SPEED);
		pos +=    tmp;
		lookat += tmp;
	}
	if ( keystate[DOWN_CAM_KEY_0] || keystate[DOWN_CAM_KEY_1] ) {
		Vector3 tmp = up*(move_speed*timer.get_dt_s()+BASE_CAM_SPEED);
		pos -=    tmp;
		lookat -= tmp;
	}
}

