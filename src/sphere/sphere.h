#ifndef __SPHERE_H__

#define __SPHERE_H__
#include "../util/vector.h"
#include "spheremodel.h"

#include <set>
#include <map>

struct vcomparer
{
	bool operator()(const Vector3& a, const Vector3& b) const
	{
		if(abs(a.x - b.x) < VECTOREPSILON)
		{
			if(abs(a.y - b.y) < VECTOREPSILON)
			{
				if(abs(a.z - b.z) < VECTOREPSILON)
					return false;
				return a.z < b.z;
			}
			return a.y < b.y;
		}
		return a.x < b.x;
	}
};

class Sphere
{
public:
	static SphereModel* get_obj(int precision);
	
private:
	Sphere(){}
	virtual ~Sphere();
	
	static std::map<int, SphereModel*> object_tab;
	
	static SphereModel* generate(int precision);
	static bool triangle_test(Vector3,Vector3,Vector3,double,double,std::map<Vector3, Vector3, vcomparer>*&);
	static void init_icosahedron(
		std::map<Vector3, Vector3, vcomparer>*&,
		std::map<Vector3, std::set<Vector3, vcomparer>, vcomparer>*&,
		std::map<Vector3, std::set<Vector3, vcomparer>, vcomparer>*&);
};

#endif //__SPHERE_H__
