#include "memory_manager.h"

#include "cuda/math.h"

#include "sphere/sphere.h"
#include "sphere/sphere_converter.h"

using namespace MEM;

MemMgr::MemMgr()
	: holder() , gpf(&holder) , ppf(&holder)
{
}

MemMgr::~MemMgr()
{
}

GLuint MemMgr::loadModels()
{
	TODO("Loading models from file");
	TODO("Return ModelInTexture sturcture insead of GLuint");

	SphereConv sc(*Sphere::get_obj(0) );

	const GLsizei size = sc.sizeTriangles();

	float*vertiecs = new float[size*3];
	float*texcoords= new float[size*2];

	sc.toTriangles( vertiecs , texcoords );

	GLuint tex;

	glGenTextures(1,&tex);
	glBindTexture(GL_TEXTURE_1D, tex);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S    , GL_CLAMP  );
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage1D(GL_TEXTURE_1D,0,GL_RGB16F,size,0,GL_RGB, GL_FLOAT , vertiecs );

	return tex;
}

void MemMgr::init()
{
	TODO("Delete or use?");
}

void MemMgr::load( const std::string& path )
{
	TODO("Loading saved points from file");

	const int size = 4096;

	// hardcoded load
	holder.resize( size );

	float3 * pos = holder.pos.map( GPU::BUF_H );
	for( int i=0 ; i<size ; i++ )
	{
		pos[i].x = (i-size/2)*2.5;
		pos[i].y = (i-size/2)*2.5;
		pos[i].z = (i-size/2)*2.5;
	}

	holder.pos.unmap();

	float  * rad = holder.radius.map( GPU::BUF_H );
	for( int i=0 ; i<size ; i++ )
		rad[i] = 1.0f;

	holder.radius.unmap();
}

void MemMgr::save( const std::string& path )
{
}

GPU::GfxPlanetFactory* MemMgr::getGfxMem()
{
	return &gpf;
}

GPU::PhxPlanetFactory* MemMgr::getPhxMem()
{
	return &ppf;
}

