#include "gfx_planet_factory.h"

#include "cuda/math.h"

using namespace MEM::MISC;

GfxPlanetFactory::GfxPlanetFactory( const PlanetHolder* holder )
	: holder(holder)
{
}

GfxPlanetFactory::~GfxPlanetFactory( )
{
}

const BufferGl<int> &GfxPlanetFactory::getModels() const
{
	return holder->model;
}

const BufferGl<float3> &GfxPlanetFactory::getLight() const
{
	return holder->light;
}

const BufferGl<int> &GfxPlanetFactory::getTexIds() const
{
	return holder->texId;
}

const BufferGl<float3>  &GfxPlanetFactory::getAtmColor () const
{
	return holder->atm_color;
}

const BufferGl<float2>  &GfxPlanetFactory::getAtmData  () const
{
	return holder->atm_data;
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

unsigned GfxPlanetFactory::size() const
{
	return holder->size();
}
