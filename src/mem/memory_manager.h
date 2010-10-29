#ifndef __MEMORY_MANAGER_H__

#define __MEMORY_MANAGER_H__

#include "gpu/gfx_planet_factory.h"
#include "gpu/phx_planet_factory.h"

namespace MEM {

	class MemMgr {
	public:
		MemMgr( );
		virtual ~MemMgr();

		GPU::GfxPlanetFactory* getGfxMem();
		GPU::PhxPlanetFactory* getPhxMem();
		
	private:
		GPU::Holder holder;

		GPU::GfxPlanetFactory gpf;
		GPU::PhxPlanetFactory ppf;
	};

}


#endif /* __MEMORY_MANAGER_H__ */

