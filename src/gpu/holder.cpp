#include "holder.h"

using namespace GPU;

PlanetHolder::PlanetHolder()
	: model(0)
	, pos(0)
	, radius(0)
	, count(0)
	, mass(0)
	, velocity(0)
{
}

PlanetHolder::~PlanetHolder()
{
}

void PlanetHolder::init( unsigned num )
{
	count.resize(1);
	count.map( BUF_H )[0] = num;
	count.unmap();
}

void PlanetHolder::resize(const size_t num)
{
	TODO("keep previous data...");
	model.resize(num);
	pos.resize(num);
	radius.resize(num);
	mass.resize(num);
	velocity.resize(num);
	count.map( BUF_H )[0] = num;
	count.unmap();
}
