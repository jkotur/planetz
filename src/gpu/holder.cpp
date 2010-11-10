#include "holder.h"

using namespace GPU;

PlanetHolder::PlanetHolder( unsigned num )
	: model(0)
	, pos(0)
	, radius(0)
	, count(1)
	, mass(0)
	, velocity(0)
{
	count.map( BUF_H )[0] = num;
	count.unmap();
}

PlanetHolder::~PlanetHolder()
{
	log_printf(INFO, "deleted planetholder\n");
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
