#ifndef __CAMERA_MANAGER_H__

#define __CAMERA_MANAGER_H__

#include <list>

#include "gfx/drawable.h"
#include "ui/input_listener.h"
#include "ui/camera.h"

namespace UI
{

class CameraMgr : public GFX::Drawable , public InputListener {
	typedef std::pair<Camera*,void*> CameraQueuePair;
	typedef std::list<CameraQueuePair> CameraQueue;

public:
	enum CAMERA_TYPES {
		FREELOOK = 0 ,
		LOCKED       ,
		ZOOMIN       ,
		CAMERA_NUM
	};

	CameraMgr(	Vector3 p = Vector3() ,
			Vector3 l = Vector3() ,
			Vector3 u = Vector3() );
	virtual ~CameraMgr();

	void init();
	void set_speed( double _s ) {}

	void request( enum CAMERA_TYPES type , void*data );
	void update ( enum CAMERA_TYPES type , void*data );

	void set_perspective(
		 const Vector3& _pos 
		,const Vector3& _lookat 
		,const Vector3& _up );

	virtual void draw() const;

	virtual void signal();
	
	virtual void on_key_down( SDLKey , Uint16 , Uint8 );
	virtual void on_mouse_motion( int x , int y );
	virtual void on_button_up( int , int , int );
	virtual bool on_button_down( int , int , int );

	boost::signal<void (float*)> sigCamChanged;
private:
	void emit_sig();
	void swap_camera();
	
	Camera * cams[CAMERA_NUM];

	float* currmat;

	Camera * currcam;

	CameraQueue next;
};

} // UI

#endif /* __CAMERA_MANAGER_H__ */

