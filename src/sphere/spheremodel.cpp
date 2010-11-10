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
	delete[] texture_points;
	delete[] texture_triangles;
}

int SphereModel::get_points_count() const
{
	return points_count;
}

const Vector3& SphereModel::get_point(int id) const
{
	return points[id];
}

const Vector3& SphereModel::get_normal(int id) const
{
	return normals[id];
}

int SphereModel::get_triangles_count() const
{
	return triangles_count;
}

const Triple& SphereModel::get_triangle(int id) const
{
	return triangles[id];
}

int SphereModel::get_texture_points_count() const
{
	return texture_points_count;
}

const Vector3& SphereModel::get_texture_point(int id) const
{
	return texture_points[id];
}

const Triple& SphereModel::get_texture_triangle(int id) const
{
	return texture_triangles[id];
}
