#include "memory_manager.h"

#include <cmath>
#include <cstring>

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

GPU::PlanetzModel MemMgr::loadModels()
{
	TODO("Memory leak: where to delete m.vertiecs and m.texCoord?");

	SphereConv sc( *Sphere::get_obj(0) );

	const GLsizei size = sc.sizeTriangles();

	float*vertiecs = new float[size*3];
	float*texcoords= new float[size*2];

	sc.toTriangles( vertiecs , texcoords );

	GPU::PlanetzModel m;

	m.len      = size;
	m.part_len = 120;

	m.parts = std::ceil((float)m.len/(float)m.part_len);

	m.vertices = new GLuint[m.parts];
	m.texCoord = new GLuint[m.parts];

	log_printf(DBG,"model: %d %d %d\n",m.len,m.part_len,m.parts);

	glGenTextures(m.parts,m.vertices);

	for( int i=0 ; i<m.parts ; i++ )
	{
		glBindTexture(GL_TEXTURE_1D, m.vertices[i] );
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S    , GL_CLAMP  );
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage1D(GL_TEXTURE_1D,0,GL_RGB16F,m.part_len,0,GL_RGB, GL_FLOAT , vertiecs+i*m.part_len*3 );
	}

	delete[]vertiecs;
	delete[]texcoords;

	return m;
}

void MemMgr::init()
{
	TODO("Delete or use?");
}

void MemMgr::load( const std::string& path )
{
	TODO("Loading saved points from file");

	const int size = 10240;

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

	uint8_t* type = holder.model.map( GPU::BUF_H );
	memset(type,0,sizeof(uint8_t)*size);
	for( int i=0 ; i<3 ; i++ )
		type[i] = 1u;
	holder.model.unmap();
}

void MemMgr::save( const std::string& path )
{
	TODO("Implement saving");
}

GPU::GfxPlanetFactory* MemMgr::getGfxMem()
{
	return &gpf;
}

GPU::PhxPlanetFactory* MemMgr::getPhxMem()
{
	return &ppf;
}

