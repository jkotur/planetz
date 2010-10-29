
#ifndef __GFX_PLANET_FACTORY_H__

#define __GFX_PLANET_FACTORY_H__

#include "buffer.h"
#include "holder.h"

namespace GPU {

	class GfxPlanet {
	public:
		GfxPlanet( int id , const Holder* h );
		virtual ~GfxPlanet();
		
		uint8_t  getModel() const;

		float3   getPosition() const;
		float    getRadius() const;
		uint32_t getCount() const;
	private:
		
	};


	class GfxPlanetFactory {
	public:
		GfxPlanetFactory( const Holder* );
		virtual ~GfxPlanetFactory( );

		const GfxPlanet getPlanet( int id ) const;

		const BufferGl<uint8_t> *getModels() const;

		const BufferGl<float3>  &getPositions() const;
		const BufferGl<float>   &getRadiuses() const;
		const BufferGl<uint32_t>&getCounts() const;

	private:

		const Holder* const holder;
	};

}

#endif /* __GFX_PLANET_FACTORY_H__ */

