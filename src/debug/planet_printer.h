#ifndef _DEBUG_PLANET_PRINTER_H_
#define _DEBUG_PLANET_PRINTER_H_

namespace MEM
{
namespace MISC
{
	class PhxPlanetFactory;
}
}

class PlanetPrinter
{
	public:
		PlanetPrinter( MEM::MISC::PhxPlanetFactory *f );
		
		void print( int id );
	private:

		MEM::MISC::PhxPlanetFactory *factory;
};

#endif // _DEBUG_PLANET_PRINTER_H_
