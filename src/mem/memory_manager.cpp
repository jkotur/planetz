#include "memory_manager.h"

#include "cuda/math.h"

using namespace MEM;

MemMgr::MemMgr()
	: holder() , gpf(&holder) , ppf(&holder)
{
}

MemMgr::~MemMgr()
{
}

void MemMgr::init()
{
	holder.init();
}

void MemMgr::load( const std::string& path )
{
	// hardcoded load
	holder.resize( 1024 );

	float3 * pos = holder.pos.map( GPU::BUF_H );
	for( int i=0 ; i<1024 ; i++ )
	{
		pos[i].x = (i-512)*2.5;
		pos[i].y = (i-512)*2.5;
		pos[i].z = (i-512)*2.5;
	}

	holder.pos.unmap();

	float  * rad = holder.radius.map( GPU::BUF_H );
	for( int i=0 ; i<1024 ; i++ )
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

