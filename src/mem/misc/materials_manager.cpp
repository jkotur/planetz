#include "materials_manager.h"

#include "debug/routines.h"

using namespace MEM;

MaterialsMgr::MaterialsMgr ()
	: id(0) , texId(0)
{
}

MaterialsMgr::~MaterialsMgr()
{
}

unsigned int MaterialsMgr::addMaterial()
{
	materials.push_back(Material());
	return id;
}

unsigned int MaterialsMgr::addMaterial( float r , float g , float b ,
		       float ke, float ka, float kd, float ks ,
		       float alpha )
{
	Material m = { r , g , b , ke , ka , kd , ks , alpha };
	materials.push_back(m);
	return id;
}

void MaterialsMgr::setColor3f( float r , float g , float b )
{
	Material*m =&materials.back();
	m->r = r;
	m->g = g;
	m->b = b;
}

void MaterialsMgr::setColor3i( int r , int g , int b )
{
	Material*m =&materials.back();
	m->r = (float)r/255.0;
	m->g = (float)g/255.0;
	m->b = (float)b/255.0;
}


void MaterialsMgr::setKe( float ke )
{
	materials.back().ke = ke;
}

void MaterialsMgr::setKa( float ka )
{
	materials.back().ka = ka;
}

void MaterialsMgr::setKd( float kd )
{
	materials.back().kd = kd;
}

void MaterialsMgr::setKs( float ks )
{
	materials.back().ks = ks;
}

void MaterialsMgr::setAlpha( float alpha )
{
	materials.back().alpha = alpha;
}

GLuint MaterialsMgr::compile()
{
	if( texId ) glDeleteTextures(1,&texId);
	++id;

	float*data = new float[ materials.size()*8 ];

	unsigned int i = 0;

	for( std::list<Material>::iterator it = materials.begin() ; it != materials.end() ; ++it )
	{
		data[i++] = it->r    ;
		data[i++] = it->g    ;
		data[i++] = it->b    ;
		data[i++] = it->alpha;
		data[i++] = it->ke   ;
		data[i++] = it->ka   ;
		data[i++] = it->kd   ;
		data[i++] = it->ks   ;
	}
	
	TODO("delete this texture somewhere");
	glGenTextures(1,&texId);

	glBindTexture(GL_TEXTURE_1D, texId );
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S    , GL_CLAMP  );
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage1D(GL_TEXTURE_1D,0,GL_RGBA16F,materials.size()*2,0,GL_RGBA, GL_FLOAT,data);
	glBindTexture(GL_TEXTURE_1D, 0 );

	delete data;
	return texId;
}

