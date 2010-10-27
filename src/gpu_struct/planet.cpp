#include "planet.h"

using namespace GPU;

Planet::Planet(uint32_t id)
	: HolderUser(id)
{
	assert( id < holder->Planet_count->h_ptr[0] );
}

Planet::~Planet()
{
}

float3 Planet::getPos() const
{
	assert( id < holder->Planet_pos->size );
	return holder->Planet_pos->h_ptr[id];
}

float Planet::getRadius() const
{
	assert( id < holder->Planet_radius->size );
	return holder->Planet_radius->h_ptr[id];
}

void Planet::setPos( float3 new_pos )
{
	assert( id < holder->Planet_pos->size );
	holder->Planet_pos->h_ptr[id] = new_pos;
//	holder->Planet_pos->fireEventContentChanged();
}

void Planet::setRadius( float new_radius )
{
	assert( id < holder->Planet_radius->size );
	holder->Planet_radius->h_ptr[id] = new_radius;
//	holder->Planet_radius->fireEventContentChanged();
}
