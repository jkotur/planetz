#include "phx_planet_factory.h"

using namespace GPU;

PhxPlanet::PhxPlanet( int id , const Holder* h )
{
}

PhxPlanet::~PhxPlanet()
{
}

uint8_t  PhxPlanet::getModel()
{
}

float3   PhxPlanet::getPosition()
{
}

float    PhxPlanet::getRadius()
{
}

uint32_t PhxPlanet::getCount()
{
}

PhxPlanetFactory::PhxPlanetFactory( const Holder* holder )
	: holder(h)
{
}

PhxPlanetFactory::~PhxPlanetFactory( )
{
}

PhxPlanet* PhxPlanetFactory::getPlanet( int id )
{
	return new PhxPlanet( id , holder );
}

BufferGl<uint8_t> *PhxPlanetFactory::getModels()
{
	return holder->planet_model;
}

BufferGl<float3>  *PhxPlanetFactory::getPositions()
{
	return holder->planet_pos;
}

BufferGl<float>   *PhxPlanetFactory::getRadiuses()
{
	return holder->planet_radius;
}

BufferGl<uint32_t>*PhxPlanetFactory::getCounts()
{
	return holder->planet_count;
}

