#ifndef __CAMERA_H__

#define __CAMERA_H__

#include <boost/signal.hpp>

#include "input/driver.h"
#include "util/vector.h"
#include "gfx/drawable.h"
#include "ui/input_listener.h"

namespace UI
{

struct CameraState
{
	Vector3 pos;
	Vector3 lookat;
	Vector3 up;
};


/** 
 * @brief obiekt odpowiedzilny za zachowanie kamery. Powinien on być
 * wyświetlony funkcją draw przed wszystkimi innymi obiektami które mają
 * być widziane z tej kamery.
 */
class Camera : public UI::InputListener {
public:
	typedef float* Matrix;

	Camera() {}
	virtual ~Camera() {}

	virtual void born( Matrix state , void*data ) =0;
	virtual void learn( void*data ) =0;
	virtual Matrix work() =0;
	virtual Matrix die () =0;
	
protected:
	Matrix matrix;
};

class CamFreeLook : public Camera {
public:
	CamFreeLook ();
	virtual ~CamFreeLook();

	virtual void born( Matrix state , void*data );
	virtual void learn( void*data );
	virtual Matrix work();
	virtual Matrix die ();

	virtual void on_mouse_motion( int x , int y ); virtual void on_button_up( int , int , int );
	virtual bool on_button_down( int , int , int );

	virtual void signal();
protected:
//        void move( float x , float y , float z );
//        void rot ( float a , float x , float y , float z );

	Vector3 pos , lookat , up;
	Vector3 right , forward;

	int ox , oy; /**< poprzednia pozycja myszy */

	double speed;
	double move_speed;

	bool rot;
};

class CamLocked : public Camera {
public:
	CamLocked ();
	virtual ~CamLocked();
	
	virtual void born( Matrix state , void*data );
	virtual void learn( void*data );
	virtual Matrix work();
	virtual Matrix die ();
	
};

class CamZoomIn : public Camera {
public:
	CamZoomIn ();
	virtual ~CamZoomIn();
	
	virtual void born( Matrix state , void*data );
	virtual void learn( void*data );
	virtual Matrix work();
	virtual Matrix die ();
	
};

} // UI

#endif /* __CAMERA_H__ */

