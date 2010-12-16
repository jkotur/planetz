#include "gfx_planet_factory.h"

#include "cuda/math.h"

using namespace MEM::MISC;

GfxPlanet::GfxPlanet( int id , const PlanetHolder* h )
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

GfxPlanetFactory::GfxPlanetFactory( const PlanetHolder* holder )
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

const BufferGl<int> &GfxPlanetFactory::getModels() const
{
	return holder->model;
}

const BufferGl<float> &GfxPlanetFactory::getEmissive() const
{
	return holder->emissive;
}

const BufferGl<int> &GfxPlanetFactory::getTexIds() const
{
	return holder->texId;
}

const BufferGl<float3>  &GfxPlanetFactory::getPositions() const
{
	return holder->pos;
}

const BufferGl<float>   &GfxPlanetFactory::getRadiuses() const
{
	return holder->radius;
}

const BufferGl<uint32_t>&GfxPlanetFactory::getCounts() const
{
	return holder->count;
}

