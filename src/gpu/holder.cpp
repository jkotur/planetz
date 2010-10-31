#include "holder.h"

using namespace GPU;

PlanetHolder::PlanetHolder(unsigned num)
	: model(num)
	, pos(num)
	, radius(num)
	, count(0) // should be 1
	, mass(num)
	, velocity(num)
{
	TODO("fix BufferGL so count can be of size 1");
	//count.map( BUF_H )[0] = num;
	//count.unmap();
}

PlanetHolder::~PlanetHolder()
{
}

void PlanetHolder::resize(const size_t num)
{
	TODO("keep previous data...");
	model.resize(num);
	pos.resize(num);
	radius.resize(num);
	mass.resize(num);
	velocity.resize(num);
	//count.map( BUF_H )[0] = num;
	//count.unmap();
}
