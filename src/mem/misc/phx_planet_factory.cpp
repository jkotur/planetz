#include "phx_planet_factory.h"

#include "cuda/math.h"

using namespace MEM::MISC;

PhxPlanet::PhxPlanet()
	: id(-1)
{
}

PhxPlanet::PhxPlanet( unsigned _id , const PlanetHolder* h )
	: id( _id )
	, holder( h )
{
	ASSERT( h );
}

PhxPlanet::~PhxPlanet()
{
}

float3   PhxPlanet::getPosition() const
{
	if( id < 0 ) return make_float3(0);
	float3 result = holder->pos.map( BUF_H )[ id ];
	holder->pos.unmap();
	return result;
}

float    PhxPlanet::getRadius() const
{
	if( id < 0 ) return 0.0f;
	float result = holder->radius.map( BUF_H )[ id ];
	holder->radius.unmap();
	return result;
}

float	PhxPlanet::getMass() const
{
	if( id < 0 ) return 0.0f;
	return holder->mass.getAt( id );
}

float3 PhxPlanet::getVelocity() const
{
	if( id < 0 ) return make_float3(0);
	return holder->velocity.getAt( id );
}

PhxPlanetFactory::PhxPlanetFactory( PlanetHolder* holder )
	: holder(holder)
{
}

PhxPlanetFactory::~PhxPlanetFactory( )
{
}

PhxPlanet PhxPlanetFactory::getPlanet( int id )
{
	return PhxPlanet( id , holder );
}

BufferGl<float3>  &PhxPlanetFactory::getPositions()
{
	return holder->pos;
}

BufferGl<float>   &PhxPlanetFactory::getRadiuses()
{
	return holder->radius;
}

BufferGl<uint32_t>&PhxPlanetFactory::getCount()
{
	return holder->count;
}

BufferCu<float> &PhxPlanetFactory::getMasses()
{
	return holder->mass;
}

BufferCu<float3> &PhxPlanetFactory::getVelocities()
{
	return holder->velocity;
}

unsigned PhxPlanetFactory::size() const
{
	return holder->size();
}

void PhxPlanetFactory::filter( BufferCu<unsigned> *mask )
{
	holder->filter( mask );
}
