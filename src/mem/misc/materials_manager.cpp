#include "materials_manager.h"

#include "debug/routines.h"

using namespace MEM::MISC;

MaterialsMgr::MaterialsMgr( Materials*mat )
	: id(0) , materials(mat)
{
}

MaterialsMgr::~MaterialsMgr()
{
}

unsigned int MaterialsMgr::addMaterial()
{
	materials->push_back(Material());
	return id++;
}

unsigned int MaterialsMgr::addMaterial(
			float r , float g , float b ,
			float ke, float ka, float kd, float ks ,
			float alpha , int texture ) 
{
	Material m = { r , g , b , ke , ka , kd , ks , alpha , texture };
	materials->push_back(m);
	return id++;
}

void MaterialsMgr::setColor3f( float r , float g , float b )
{
	Material*m =&materials->back();
	m->r = r;
	m->g = g;
	m->b = b;
}

void MaterialsMgr::setColor3i( int r , int g , int b )
{
	Material*m =&materials->back();
	m->r = (float)r/255.0;
	m->g = (float)g/255.0;
	m->b = (float)b/255.0;
}

void MaterialsMgr::setKe( float ke )
{
	materials->back().ke = ke;
}

void MaterialsMgr::setKa( float ka )
{
	materials->back().ka = ka;
}

void MaterialsMgr::setKd( float kd )
{
	materials->back().kd = kd;
}

void MaterialsMgr::setKs( float ks )
{
	materials->back().ks = ks;
}

void MaterialsMgr::setAlpha( float alpha )
{
	materials->back().alpha = alpha;
}

void MaterialsMgr::setTexture( int texture )
{
	materials->back().texture = texture;
}

