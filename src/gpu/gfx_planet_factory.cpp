#include "gfx_planet_factory.h"

#include "cuda/math.h"

using namespace GPU;

GfxPlanet::GfxPlanet( int id , const Holder* h )
{
}

GfxPlanet::~GfxPlanet()
{
}

uint8_t  GfxPlanet::getModel() const
{
	return 0;
}

float3   GfxPlanet::getPosition() const
{
	return make_float3(0,0,0);
}

float    GfxPlanet::getRadius() const
{
	return 0;
}

uint32_t GfxPlanet::getCount() const
{
	return 0;
}

GfxPlanetFactory::GfxPlanetFactory( const Holder* holder )
	: holder(holder)
{
}

GfxPlanetFactory::~GfxPlanetFactory( )
{
}

const GfxPlanet GfxPlanetFactory::getPlanet( int id ) const
{
	return GfxPlanet( id , holder );
}

const BufferGl<uint8_t> *GfxPlanetFactory::getModels() const
{
	return &holder->planet_model;
}

const BufferGl<float3>  &GfxPlanetFactory::getPositions() const
{
	return holder->planet_pos;
}

const BufferGl<float>   &GfxPlanetFactory::getRadiuses() const
{
	return holder->planet_radius;
}

const BufferGl<uint32_t>&GfxPlanetFactory::getCounts() const
{
	return holder->planet_count;
}

