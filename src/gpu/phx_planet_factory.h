
#ifndef __PHX_PLANET_FACTORY_H__

#define __PHX_PLANET_FACTORY_H__

#include "buffer.h"
#include "holder.h"

namespace GPU {

	class PhxPlanet {
	public:
		PhxPlanet( int id , const Holder* h );
		virtual ~PhxPlanet();
		
		float3   getPosition();
		float    getRadius();
		uint32_t getCount();
	private:
		
	};


	class PhxPlanetFactory {
	public:
		PhxPlanetFactory( Holder* );
		virtual ~PhxPlanetFactory( );

		PhxPlanet getPlanet( int id );

		BufferGl<uint8_t> &getModels();

		BufferGl<float3>  &getPositions();
		BufferGl<float>   &getRadiuses();
		BufferGl<uint32_t>&getCounts();

	private:

		Holder* const holder;
	};

}

#endif /* __PHX_PLANET_FACTORY_H__ */

