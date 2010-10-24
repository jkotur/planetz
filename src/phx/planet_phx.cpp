#include "planet.h"
#include "../util/logger.h"

using namespace Phx;


Planet::Planet( const Vector3& _p , const Vector3& _s , double _m, double _r)
	: mdel(false) , pos(_p) , speed(_s) , m(_m) , radius(_r)
{
}

Planet::~Planet()
{
}

