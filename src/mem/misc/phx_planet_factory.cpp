#include "phx_planet_factory.h"

#include "cuda/math.h"

using namespace MEM::MISC;

PhxPlanet::PhxPlanet( unsigned _id , const PlanetHolder* h )
	: id( _id )
	, holder( h )
{
	assert( h );
}

PhxPlanet::~PhxPlanet()
{
	log_printf(DBG,"[DEL] PhxPlanet\n");
}

float3   PhxPlanet::getPosition() const
{
	float3 result = holder->pos.map( BUF_H )[ id ];
	holder->pos.unmap(); // FIXME: what if it was already mapped?
	return result;
}

float    PhxPlanet::getRadius() const
{
	float result = holder->radius.map( BUF_H )[ id ];
	holder->radius.unmap(); // FIXME as above
	return result;
}

float	PhxPlanet::getMass() const
{
	return .0f;// holder->mass.
}

float3 PhxPlanet::getVelocity() const
{
	return make_float3(.0f, .0f, .0f);
}

PhxPlanetFactory::PhxPlanetFactory( PlanetHolder* holder )
	: holder(holder)
{
	log_printf(DBG, "TESTING: pos addr = %x\n", this );
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
