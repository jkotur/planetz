
#ifndef __PHX_PLANET_FACTORY_H__

#define __PHX_PLANET_FACTORY_H__

#include "buffer.h"
#include "holder.h"
#include <boost/shared_ptr.hpp>

namespace MEM
{
namespace MISC
{
	class PhxPlanet
	{
	public:
		PhxPlanet( unsigned id , const PlanetHolder* h );
		virtual ~PhxPlanet();
		
		float3  getPosition() const;
		float   getRadius() const;
		float	getMass() const;
		float3	getVelocity() const;

	private:
		unsigned id;
		const PlanetHolder* holder;
	};


	class PhxPlanetFactory
	{
	public:
		PhxPlanetFactory( PlanetHolder* );
		virtual ~PhxPlanetFactory( );

		PhxPlanet getPlanet( int id );

		BufferGl<float3>  &getPositions();
		BufferGl<float>   &getRadiuses();
		BufferCu<float>   &getMasses();
		BufferCu<float3>  &getVelocities();

		BufferGl<uint32_t>&getCount();

		unsigned size() const;

	private:

		PlanetHolder* const holder;
	};

}
}
#endif /* __PHX_PLANET_FACTORY_H__ */
