#ifndef __PLANETZ_SETTER_H__

#define __PLANETZ_SETTER_H__

#include <GL/glew.h>
#include <boost/signal.hpp>

#include "ui/input_listener.h"
#include "gfx/drawable.h"
#include "util/vector.h"
#include "mem/misc/planet_params.h"

namespace UI {
class PlanetzSetter : public InputListener , public GFX::Drawable {
	enum MODE {
		MODE_NONE,
		MODE_POS,
		MODE_RADIUS,
		MODE_VEL,
		MODE_DRAW,
		MODE_COUNT,
		MODE_POS_ONLY,
		MODE_RADIUS_ONLY,
		MODE_VEL_ONLY,
	};
public:
	PlanetzSetter ();
	virtual ~PlanetzSetter();
	
	virtual void draw() const;

	virtual void signal();
	
	virtual void on_mouse_motion( int x , int y );
	virtual bool on_button_down( int , int , int );

	void changePos()
	{	 mode = MODE_POS_ONLY; }
	void changeVel()
	{	 mode = MODE_VEL_ONLY; }
	void changeRadius()
	{	mode = MODE_RADIUS_ONLY; }
	void change( const MEM::MISC::PlanetParams& pp );

	void clear()
	{	mode = MODE_NONE; }

	void update( const MEM::MISC::PlanetParams& pp );

	boost::signal<void (const MEM::MISC::PlanetParams& pp)> on_planet_changed;
private:
	void drawVel() const;
	void drawRadius() const;

	Vector3 screen_camera( int x , int y , float z );
	Vector3 camera_world( const Vector3& in );

	void inverse( GLfloat dst[16] , const GLfloat src[16] );
	Vector3 position , velocity;
	Vector3 position_mv;
	float radius;
	enum MODE mode;

	GLUquadric* quad;

	float Z;
	
};
} // UI 

#endif /* __PLANETZ_SETTER_H__ */

