#include "memory_manager.h"

#include <cmath>
#include <cstring>

#include "cuda/math.h"

#include "misc/materials_manager.h"

using namespace MEM;

MemMgr::MemMgr()
	: matTexId(0) , texTexId(0) , holder() , gpf(&holder) , ppf(&holder)
{
}

MemMgr::~MemMgr()
{
	if( matTexId ) glDeleteTextures(1,&matTexId);
	if( texTexId ) glDeleteTextures(1,&texTexId);
}

GLuint MemMgr::loadTextures( const MISC::Textures& ctex )
{
	TODO("Make this more F2C-C2G-like");

	if( texTexId ) glDeleteTextures(1,&texTexId);

	const GLint TEX_W = 1024;
	const GLint TEX_H = 512;
	GLuint texNum = ctex.size();

	glPixelStorei(GL_UNPACK_ALIGNMENT,4);

	glGenTextures(1,&texTexId);
	glBindTexture( GL_TEXTURE_2D_ARRAY , texTexId );

	glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
	glTexImage3D(GL_TEXTURE_2D_ARRAY,0,GL_RGB,TEX_W,TEX_H,texNum,0,GL_RGB,GL_UNSIGNED_BYTE,NULL);

	int z = 0;
	for( MISC::Textures::const_iterator i = ctex.begin() ; i != ctex.end() ; ++i , z++ )
	{
		ASSERT_MSG( TEX_W == (*i)->w && TEX_H == (*i)->h ,
				"Texture must be %dx%d" , TEX_W , TEX_H );

		GLenum format = (**i).format->Amask ? GL_RGBA : GL_RGB ;

		glTexSubImage3D(GL_TEXTURE_2D_ARRAY,0,0,0,z,TEX_W,TEX_H,1,format,GL_UNSIGNED_BYTE,(**i).pixels);
	}

	glBindTexture( GL_TEXTURE_2D_ARRAY , 0 );

	return texTexId;
}

GLuint MemMgr::loadMaterials( const MISC::Materials& materials )
{
	if( matTexId ) glDeleteTextures(1,&matTexId);

	float*data = new float[ materials.size()*8 ];

	unsigned int i = 0;

	for( MISC::Materials::const_iterator it = materials.begin() ; it != materials.end() ; ++it )
	{
		// first
		data[i++] = it->r    ;
		data[i++] = it->g    ;
		data[i++] = it->b    ;
		data[i++] = it->ke   ;
		// second
		data[i++] = it->ka   ;
		data[i++] = it->kd   ;
		data[i++] = it->ks   ;
		data[i++] = it->alpha;
	}
	
	glGenTextures(1,&matTexId);

	glBindTexture(GL_TEXTURE_1D, matTexId );
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S    , GL_CLAMP  );
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage1D(GL_TEXTURE_1D,0,GL_RGBA16F,materials.size()*2,0,GL_RGBA, GL_FLOAT,data);
	glBindTexture(GL_TEXTURE_1D, 0 );

	delete data;
	return matTexId;
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
		type[i] = pl->model[i] * 2; // model must be even
	holder.model.unmap();

	holder.emissive.bind();
	float *em = holder.emissive.map( MISC::BUF_H );
	for( unsigned i=0; i<size; ++i )
		em[i] = pl->emissive[i];
	holder.emissive.unmap();

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

