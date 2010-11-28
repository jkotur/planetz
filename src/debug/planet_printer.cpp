#include <sstream>
#include "planet_printer.h"
#include "gfx/planetz_picker.h"
#include "mem/misc/phx_planet_factory.h"

PlanetPrinter::PlanetPrinter( MEM::MISC::PhxPlanetFactory *f, GFX::PlanetzPicker *pp )
	: factory( f )
	, picker( pp )
{}

bool PlanetPrinter::on_button_down( int button, int x, int y )
{
	picker->render( x, y );
	int id = picker->getId();
	if( -1 != id )
	{
		print( id );
	}
	return false;
}

void PlanetPrinter::print( int id )
{
	MEM::MISC::PhxPlanet p = factory->getPlanet( id );
	std::stringstream ss;
	ss << "Picked planet " << id << std::endl;
	float3 pos = p.getPosition();
	ss << "Position: [" << pos.x << ", " << pos.y << ", " << pos.z << "]\n";
	ss << "Radius: " << p.getRadius() << std::endl;
	ss << "Mass: " << p.getMass() << std::endl;
	float3 v = p.getVelocity();
	ss << "Velocity: [" << v.x << ", " << v.y << ", " << v.z << "]\n";
	log_printf(INFO, ss.str().c_str());
}
