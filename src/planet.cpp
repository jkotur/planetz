#include "planet.h"

Planet::Planet( Gfx::Planet*gp , Phx::Planet*lp )
	: mdel(false) , gfx_obj(gp) , phx_obj(lp)
{
}

Planet::~Planet()
{
	/** usuwanie obiektu z modelu fizycznego
	 * gdy model planety zostanie skasowany
	 * inaczej niz poprzez fizyke gry */
//        if( !phx_obj->deleted() )
//                phx_model->erase( phx_obj );
	if( gfx_obj ) delete gfx_obj;
	if( phx_obj ) delete phx_obj; 
}

void Planet::render()
{
	if( gfx_obj->selected() ) {
		Vector3 surface = phx_obj->get_velocity();
		surface.normalize();
		surface*=phx_obj->get_radius();
		arr.render( phx_obj->get_pos() + surface , phx_obj->get_velocity() );
	}

	gfx_obj->render( phx_obj->get_pos() , phx_obj->get_radius() );
}

