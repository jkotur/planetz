#ifndef __CAMERA_H__

#define __CAMERA_H__

#include "./util/vector.h"
#include "./input/driver.h"

class Camera {
public:
	Camera ( const Vector3& pos 
		,const Vector3& lookat 
		,const Vector3& up );
	virtual ~Camera();

	void on_key_down( int k );
	void on_mouse_motion( int x , int y );
	void on_button_up( int , int , int );
	bool on_button_down( int , int , int );
	void gl_lookat();
	void signal();

	void set_perspective(
		 const Vector3& _pos 
		,const Vector3& _lookat 
		,const Vector3& _up )
	{
		pos = _pos;
		lookat = _lookat;
		up = _up;

		init();
	}


	void set_speed( double _s )
	{	move_speed = _s; }

	Vector3 get_pos() const
	{	return pos; }

	Vector3 get_lookat() const 
	{	return lookat; }

	Vector3 get_up() const 
	{	return up; }

//        void set_joy( CLocationDriver * _j )
//        {	joy = _j; }
private:
	void init();
	
	Vector3 pos , lookat , up;
	Vector3 right , forward;

	int ox , oy; /**< poprzednia pozycja myszy */

	double speed;
	double move_speed;

	bool rot;

//        CLocationDriver * joy;
};


#endif /* __CAMERA_H__ */

