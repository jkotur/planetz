#ifndef __SPHERE_CONVERTER_H__

#define __SPHERE_CONVERTER_H__

#include "spheremodel.h"

class SphereConv {
public:
	SphereConv( const SphereModel& sm ) : sm(sm) {}
	virtual ~SphereConv() {}

	void toTriangleStrip( float*vert , float*texCoord );
	void toTriangles( float*vert , float*texCoord );

	unsigned sizeTriangles()
	{	return sm.get_triangles_count()*3; }
	unsigned sizeTriangleStrip()
	{	return (sm.get_points_count() + 1)*3; }
private:

	const SphereModel& sm;
};


#endif /* __SPHERE_CONVERTER_H__ */

