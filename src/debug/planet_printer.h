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
		
		bool on_button_down( int b, int x, int y );

	private:
		void print( int id );

		MEM::MISC::PhxPlanetFactory *factory;
		GFX::PlanetzPicker *picker;
};

#endif // _DEBUG_PLANET_PRINTER_H_
