#include "memory_manager.h"

#include <cmath>
#include <cstring>

#include "cuda/math.h"

#include "sphere/sphere.h"
#include "sphere/sphere_converter.h"

#include "misc/materials_manager.h"

using namespace MEM;

MemMgr::MemMgr()
	: holder() , gpf(&holder) , ppf(&holder)
{
}

MemMgr::~MemMgr()
{
}

MISC::PlanetzModel MemMgr::loadModels()
{
	TODO("Memory leak: where to delete m.vertiecs and m.texCoord?");

	SphereConv sc( *Sphere::get_obj(0) );

	const GLsizei size = sc.sizeTriangles();

	float*vertiecs = new float[size*3];
	float*texcoords= new float[size*2];

	sc.toTriangles( vertiecs , texcoords );

	MISC::PlanetzModel m;

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

GLuint MemMgr::loadMaterials()
{
	TODO("Load materials from file or sth");

	MaterialsMgr mgr;

	               // r    g    b   ke   ka   kd    ks  alpha
	mgr.addMaterial( .5 , .1 , .0 , .0 , .2 , 1.5 , 0 , 1 );
	mgr.addMaterial( .0 , .3 , 1. , .0 , .2 , 1.5 , 0 , 1 );
	mgr.addMaterial( .5 , 1. , .0 , .0 , .2 , 1.5 , 0 , 1 );
	mgr.addMaterial( 1. , 1. , 1. , .8 , .2 , 1.0 , 0 , 1 );

	return mgr.compile();
}

void MemMgr::setPlanets( MISC::CpuPlanetHolder *pl )
{
	size_t size = pl->size();

	holder.resize( size );

	float3 * pos = holder.pos.map( MISC::BUF_H );
	for( unsigned i=0 ; i<size ; i++ )
		pos[i] = pl->pos[i];
	holder.pos.unmap();

	float  * rad = holder.radius.map( MISC::BUF_H );
	for( unsigned i=0 ; i<size ; i++ )
		rad[i] = pl->radius[i];
	holder.radius.unmap();

	int* type = holder.model.map( MISC::BUF_H );
	for( unsigned i=0 ; i<size ; i++ )
		if( !(i % 32) && i<32*8 ) 
			type[i] = 3*2;
		else
			type[i] = pl->model[i] * 2; // model must be even
	holder.model.unmap();

	holder.count.assign( pl->count[0] );

	holder.mass.bind();
	float* mass = holder.mass.h_data();
	for( unsigned i=0; i<size; ++i )
		mass[i] = pl->mass[i];
	holder.mass.unbind();

	holder.velocity.bind();
	float3 *vel = holder.velocity.h_data();
	for( unsigned i=0; i<size; ++i )
		vel[i] = pl->velocity[i];
	holder.velocity.unbind();
}

MISC::CpuPlanetHolder *MemMgr::getPlanets()
{
	size_t size = holder.size();
	MISC::CpuPlanetHolder *pl = new MISC::CpuPlanetHolder( size );

	if( !size )
		return pl;

	float3 * pos = holder.pos.map( MISC::BUF_H );
	for( unsigned i=0 ; i<size ; i++ )
		pl->pos[i] = pos[i];
	holder.pos.unmap();

	float  * rad = holder.radius.map( MISC::BUF_H );
	for( unsigned i=0 ; i<size ; i++ )
		pl->radius[i] = rad[i];
	holder.radius.unmap();

	int* type = holder.model.map( MISC::BUF_H );
	for( unsigned i=0 ; i<size ; i++ )
		pl->model[i] = type[i] / 2; // model must've been even :O
	holder.model.unmap();

	holder.count.assign( pl->count[0] );

	holder.mass.bind();
	float* mass = holder.mass.h_data();
	for( unsigned i=0; i<size; ++i )
		pl->mass[i] = mass[i];
	holder.mass.unbind();

	holder.velocity.bind();
	float3 *vel = holder.velocity.h_data();
	for( unsigned i=0; i<size; ++i )
		pl->velocity[i] = vel[i];
	holder.velocity.unbind();

	return pl;
}

MISC::GfxPlanetFactory* MemMgr::getGfxMem()
{
	return &gpf;
}

MISC::PhxPlanetFactory* MemMgr::getPhxMem()
{
	return &ppf;
}

