#include "memory_manager.h"

using namespace MEM;

MemMgr::MemMgr()
	: holder() , gpf(&holder) , ppf(&holder)
{
}

MemMgr::~MemMgr()
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

