#ifndef _DEBUG_PLANET_PRINTER_H_
#define _DEBUG_PLANET_PRINTER_H_

namespace MEM
{
namespace MISC
{
	class PhxPlanetFactory;
}
}

namespace GFX
{
	class PlanetzPicker;
}

class PlanetPrinter
{
	public:
		PlanetPrinter( MEM::MISC::PhxPlanetFactory *f, GFX::PlanetzPicker *pp );
		
		void print( int id );
	private:

		MEM::MISC::PhxPlanetFactory *factory;
		GFX::PlanetzPicker *picker;
};

#endif // _DEBUG_PLANET_PRINTER_H_
