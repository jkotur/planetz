#ifndef __MEMORY_MANAGER_H__

#define __MEMORY_MANAGER_H__

#include <string>

#include "misc/gfx_planet_factory.h"
#include "misc/phx_planet_factory.h"
#include "misc/planet_model.h"

namespace MEM {

	class MemMgr {
	public:
		MemMgr( );
		virtual ~MemMgr();

		void init();

		MISC::GfxPlanetFactory* getGfxMem();
		MISC::PhxPlanetFactory* getPhxMem();
		
		MISC::PlanetzModel loadModels();

		void load( const std::string& path );
		void save( const std::string& path );
	private:
		MISC::PlanetHolder holder;

		MISC::GfxPlanetFactory gpf;
		MISC::PhxPlanetFactory ppf;
	};

}


#endif /* __MEMORY_MANAGER_H__ */

