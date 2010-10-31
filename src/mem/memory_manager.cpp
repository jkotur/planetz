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

void MemMgr::load( const std::string& path )
{
	// hardcoded load
	holder.resize( 10 );

	float3 * pos = holder.pos.map( GPU::BUF_H );
	for( int i=0 ; i<10 ; i++ )
	{
		pos[i].x = i-5;
		pos[i].y = i-5;
		pos[i].z = i-5;
	}

	holder.pos.unmap();

	float  * rad = holder.radius.map( GPU::BUF_H );
	for( int i=0 ; i<10 ; i++ )
		rad[i] = 1.0f * i;

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

