#pragma once
#include "../util/vector.h"

struct Triple
{
	int p1, p2, p3;
	Triple(int p1, int p2, int p3): p1(p1), p2(p2), p3(p3) {}
	Triple(){}
};

class SphereModel
{
private:
	int points_count;
	Vector3* points;
	Vector3* normals;
	
	int triangles_count;
	Triple* triangles;
	
	int texture_points_count;
	Vector3* texture_points;
	Triple* texture_triangles;
	
	SphereModel(int, int, int);
	~SphereModel();
public:
	int get_points_count();
	const Vector3& get_point(int);
	const Vector3& get_normal(int);
	int get_triangles_count();
	const Triple& get_triangle(int);
	int get_texture_points_count();
	const Vector3& get_texture_point(int);
	const Triple& get_texture_triangle(int);
	
	friend class Sphere;
};
