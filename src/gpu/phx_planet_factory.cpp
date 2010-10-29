#include "phx_planet_factory.h"

#include "cuda/math.h"

using namespace GPU;

PhxPlanet::PhxPlanet( int id , const Holder* h )
{
}

PhxPlanet::~PhxPlanet()
{
}

float3   PhxPlanet::getPosition()
{
	return make_float3(0,0,0);
}

float    PhxPlanet::getRadius()
{
	return .0f;
}

uint32_t PhxPlanet::getCount()
{
	return 0;
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

BufferGl<uint32_t>&PhxPlanetFactory::getCounts()
{
	return holder->planet_count;
}

