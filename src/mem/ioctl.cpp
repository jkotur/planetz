#include "ioctl.h"
#include "saver.h"

using namespace MEM;

class IOCtl::Impl
{
	public:
		void save( PhxPlanetFactory*, GfxPlanetFactory*, const std::string &path );
		void load( PhxPlanetFactory*, GfxPlanetFactory*, const std::string &path );

	private:
		Saver s;
		
};

IOCtl::IOCtl()
{
	impl = new IOCtl::Impl();
}

IOCtl::~IOCtl()
{
	delete impl;
}

void IOCtl::save( PhxPlanetFactory* planets_phx, GfxPlanetFactory *planets_gfx, const std::string &path )
{
	impl->save( planets_phx, planets_gfx, path );
}

void IOCtl::load( PhxPlanetFactory* planets_phx, GfxPlanetFactory *planets_gfx, const std::string &path )
{
	impl->load( planets_phx, planets_gfx, path );
}

void IOCtl::save( PhxPlanetFactory* planets_phx, GfxPlanetFactory *planets_gfx )
{
	save( planets_phx, planets_gfx, DATA("saves/first.sav") );
}

void IOCtl::load( PhxPlanetFactory* planets_phx, GfxPlanetFactory *planets_gfx )
{
	load( planets_phx, planets_gfx, DATA("saves/first.sav") );
}

void IOCtl::Impl::save( PhxPlanetFactory* planets_phx, GfxPlanetFactory *planets_gfx )
{
	s.save( planets_phx, planets_gfx );
}

void IOCtl::Impl::load( PhxPlanetFactory* planets_phx, GfxPlanetFactory *planets_gfx )
{
	s.load( planets_phx, planets_gfx );
}
