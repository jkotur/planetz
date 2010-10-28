#include "gfx_planet_factory.h"

using namespace GPU;

GfxPlanet::GfxPlanet( int id , const Holder* h )
{
}

GfxPlanet::~GfxPlanet()
{
}

uint8_t  GfxPlanet::getModel() const
{
}

float3   GfxPlanet::getPosition() const
{
}

float    GfxPlanet::getRadius() const
{
}

uint32_t GfxPlanet::getCount() const
{
}

GfxPlanetFactory::GfxPlanetFactory( const Holder* holder )
	: holder(h)
{
}

GfxPlanetFactory::~GfxPlanetFactory( )
{
}

const GfxPlanet* GfxPlanetFactory::getPlanet( int id ) const
{
	return new GfxPlanet( id , holder );
}

BufferGl<uint8_t> *GfxPlanetFactory::getModels() const
{
	return holder->planet_model;
}

BufferGl<float3>  *GfxPlanetFactory::getPositions() const
{
	return holder->planet_pos;
}

BufferGl<float>   *GfxPlanetFactory::getRadiuses() const
{
	return holder->planet_radius;
}

BufferGl<uint32_t>*GfxPlanetFactory::getCounts() const
{
	return holder->planet_count;
}

