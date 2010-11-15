#ifndef __SAVER_H__

#define __SAVER_H__

#include <string>

namespace MEM
{
	namespace MISC
	{
		class PhxPlanetFactory;
		class GfxPlanetFactory;
	}

	class Saver {
	public:
		Saver();
		virtual ~Saver();

		void save( MEM::MISC::PhxPlanetFactory* planets_phx, MEM::MISC::GfxPlanetFactory* planets_gfx, const std::string& path );
		void load( MEM::MISC::PhxPlanetFactory* planets_phx, MEM::MISC::GfxPlanetFactory* planets_gfx, const std::string& path );

	private:
		void map_buffers( MEM::MISC::PhxPlanetFactory *planets_phx, MEM::MISC::GfxPlanetFactory *planets_gfx);
		void unmap_buffers( MEM::MISC::PhxPlanetFactory *planets_phx, MEM::MISC::GfxPlanetFactory *planets_gfx);
	};
}

#endif /* __SAVER_H__ */

