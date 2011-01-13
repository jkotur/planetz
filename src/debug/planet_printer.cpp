#include <sstream>
#include <cmath>
#include "planet_printer.h"
#include "mem/misc/phx_planet_factory.h"

PlanetPrinter::PlanetPrinter( MEM::MISC::PhxPlanetFactory *f )
	: factory( f )
{}

void PlanetPrinter::print( int id )
{
	if( id == -1 ) return;
	MEM::MISC::PhxPlanet p = factory->getPlanet( id );
	std::stringstream ss;
	ss << "Picked planet " << id << std::endl;
	float3 pos = p.getPosition();
	ss << "Position: [" << pos.x << ", " << pos.y << ", " << pos.z << "]\n";
	ss << "Radius: " << p.getRadius() << std::endl;
	ss << "Mass: " << p.getMass() << std::endl;
	float3 v = p.getVelocity();
	float vlen = sqrt( v.x * v.x + v.y * v.y + v.z * v.z );
	ss << "Velocity: [" << v.x << ", " << v.y << ", " << v.z << "] " << vlen << "\n";
	log_printf(INFO, ss.str().c_str());
}
