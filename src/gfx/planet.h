#ifndef __GFX_PLANET_H__

#define __GFX_PLANET_H__

#include <GL/glew.h>

#include "../util/vector.h"
#include "../util/animation.h"

#include "../sphere/sphere.h"

#include "../gfx/texture.h"

namespace GFX {

class Planet {
public:
	Planet ( int detals = 1 );
	virtual ~Planet();

	void render( const Vector3& pos , double radius );

	int get_id()
	{	return id; }

	void select()
	{	selected_b =true; }
	void deselect()
	{	selected_b =false; }
	bool selected()
	{	return selected_b; }
private:
	void set_ax( double _ax )
	{	a_x = _ax; }

	GLint l;
	int id;
	bool selected_b;

	double a_x;

	int details;

	float sel_col[3];
	float col[3];

	Texture* tex;
	Animation<double> rot_x;

	static int count;
};

}

#endif /* __GFX_PLANET_H__ */

