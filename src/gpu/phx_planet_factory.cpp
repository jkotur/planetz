#include "phx_planet_factory.h"

#include "cuda/math.h"

using namespace GPU;

PhxPlanet::PhxPlanet( unsigned _id , const Holder* h )
	: id( _id )
	, holder( h )
{
	assert( h );
}

PhxPlanet::~PhxPlanet()
{
}

float3   PhxPlanet::getPosition() const
{
	float3 result = holder->planet_pos.map( BUF_H )[ id ];
	holder->planet_pos.unmap(); // FIXME: what if it was already mapped?
	return result;
}

float    PhxPlanet::getRadius() const
{
	float result = holder->planet_radius.map( BUF_H )[ id ];
	holder->planet_radius.unmap(); // FIXME as above
	return result;
}

float	PhxPlanet::getMass() const
{
	return .0f;// holder->planet_mass.
}

float3 PhxPlanet::getVelocity() const
{
	return make_float3(.0f, .0f, .0f);
}

PhxPlanetFactory::PhxPlanetFactory( Holder* holder )
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

BufferGl<uint8_t> &PhxPlanetFactory::getModels()
{
	return holder->planet_model;
}

BufferGl<float3>  &PhxPlanetFactory::getPositions()
{
	return holder->planet_pos;
}

BufferGl<float>   &PhxPlanetFactory::getRadiuses()
{
	return holder->planet_radius;
}

BufferGl<uint32_t>&PhxPlanetFactory::getCount()
{
	return holder->planet_count;
}

