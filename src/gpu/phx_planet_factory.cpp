#include "phx_planet_factory.h"

#include "cuda/math.h"

using namespace GPU;

PhxPlanet::PhxPlanet( unsigned _id , const PlanetHolder* h )
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
	return holder->model;
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

