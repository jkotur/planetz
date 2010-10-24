#include "spheremodel.h"
	
SphereModel::SphereModel(int pc, int tc, int tpc)
{
	points_count = pc;
	triangles_count = tc;
	texture_points_count = tpc;
	points = new Vector3[pc];
	normals = new Vector3[pc];
	triangles = new Triple[tc];
	texture_triangles = new Triple[tc];
	texture_points = new Vector3[tpc];
}

SphereModel::~SphereModel()
{
	delete[] points;
	delete[] normals;
	delete[] triangles;
	delete[]	texture_points;
	delete[] texture_triangles;
}

int SphereModel::get_points_count()
{
	return points_count;
}

const Vector3& SphereModel::get_point(int id)
{
	return points[id];
}

const Vector3& SphereModel::get_normal(int id)
{
	return normals[id];
}

int SphereModel::get_triangles_count()
{
	return triangles_count;
}

const Triple& SphereModel::get_triangle(int id)
{
	return triangles[id];
}

int SphereModel::get_texture_points_count()
{
	return texture_points_count;
}

const Vector3& SphereModel::get_texture_point(int id)
{
	return texture_points[id];
}

const Triple& SphereModel::get_texture_triangle(int id)
{
	return texture_triangles[id];
}
