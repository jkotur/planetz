#ifndef __MEM_MISC_PLANET_PARAMS_H__
#define __MEM_MISC_PLANET_PARAMS_H__

namespace MEM
{
namespace MISC
{
struct PlanetParams
{
	float3 pos;
	float3 vel;
	float mass;
	float radius;
	int model;
	PlanetParams() {}
	PlanetParams( float3 _pos, float3 _vel, float _mass, float _radius, int _model )
		: pos( _pos )
		, vel( _vel )
		, mass( _mass )
		, radius( _radius )
		, model( _model )
	{}
};
}
}

#endif // __MEM_MISC_PLANET_PARAMS_H__
