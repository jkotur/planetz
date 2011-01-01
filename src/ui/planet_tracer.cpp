#include "planet_tracer.h"
#include "gfx/planetz_picker.h"
#include "mem/misc/phx_planet_factory.h"
#include "camera.h"

PlanetTracer::PlanetTracer( MEM::MISC::PhxPlanetFactory *f, GFX::PlanetzPicker *pp, Camera *cam )
	: factory( f )
	, picker( pp )
	, camera( cam )
	, id(-1)
{
}

PlanetTracer::~PlanetTracer()
{
}

bool PlanetTracer::on_button_down( int button, int x, int y )
{
	if( 1 != button )
		return false;
	picker->render( x, y );
	id = picker->getId();
	if( -1 != id )
	{
		MEM::MISC::PhxPlanet p = factory->getPlanet( id );
		float3 pos = p.getPosition();
		dist = camera->get_pos() - Vector3( pos.x, pos.y, pos.z );
	}
	return false;
}

void PlanetTracer::refresh()
{
	if( -1 == id )
		return;
	MEM::MISC::PhxPlanet p = factory->getPlanet( id );
	float3 f3pos = p.getPosition();
	Vector3 planet_pos( f3pos.x, f3pos.y, f3pos.z );
	Vector3 cam_pos = planet_pos + dist;
	camera->set_perspective( cam_pos, camera->get_lookat(), camera->get_up() );
}
