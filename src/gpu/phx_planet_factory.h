
#ifndef __PHX_PLANET_FACTORY_H__

#define __PHX_PLANET_FACTORY_H__

#include "buffer.h"
#include "holder.h"
#include <boost/shared_ptr.hpp>

namespace GPU {

	class PhxPlanet {
	public:
		PhxPlanet( unsigned id , const Holder* h );
		virtual ~PhxPlanet();
		
		float3  getPosition() const;
		float   getRadius() const;
		float	getMass() const;
		float3	getVelocity() const;

	private:
		unsigned id;
		const Holder* holder;
	};


	class PhxPlanetFactory {
	public:
		PhxPlanetFactory( Holder* );
		virtual ~PhxPlanetFactory( );

		PhxPlanet getPlanet( int id );

		BufferGl<uint8_t> &getModels();

		BufferGl<float3>  &getPositions();
		BufferGl<float>   &getRadiuses();
		BufferGl<uint32_t>&getCount();

	private:

		Holder* const holder;
	};

}

#endif /* __PHX_PLANET_FACTORY_H__ */

