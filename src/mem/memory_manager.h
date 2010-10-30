#ifndef __MEMORY_MANAGER_H__

#define __MEMORY_MANAGER_H__

#include <string>

#include "gpu/gfx_planet_factory.h"
#include "gpu/phx_planet_factory.h"

namespace MEM {

	class MemMgr {
	public:
		MemMgr( );
		virtual ~MemMgr();

		GPU::GfxPlanetFactory* getGfxMem();
		GPU::PhxPlanetFactory* getPhxMem();
		
		void load( const std::string& path );
		void save( const std::string& path );
	private:
		GPU::Holder holder;

		GPU::GfxPlanetFactory gpf;
		GPU::PhxPlanetFactory ppf;
	};

}


#endif /* __MEMORY_MANAGER_H__ */

