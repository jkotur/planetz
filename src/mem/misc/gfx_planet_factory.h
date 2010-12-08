
#ifndef __GFX_PLANET_FACTORY_H__

#define __GFX_PLANET_FACTORY_H__

#include "buffer.h"
#include "holder.h"

namespace MEM
{
namespace MISC
{

	class GfxPlanet {
	public:
		GfxPlanet( int id , const PlanetHolder* h );
		virtual ~GfxPlanet();
		
		uint8_t  getModel() const;

		float3   getPosition() const;
		float    getRadius() const;
		uint32_t getCount() const;
	private:
		
	};


	class GfxPlanetFactory {
	public:
		GfxPlanetFactory( const PlanetHolder* );
		virtual ~GfxPlanetFactory( );

		const GfxPlanet getPlanet( int id ) const;

		const BufferGl<int>     &getModels() const;
		const BufferGl<float>   &getEmissive() const;

		const BufferGl<float3>  &getPositions() const;
		const BufferGl<float>   &getRadiuses() const;
		const BufferGl<uint32_t>&getCounts() const;

	private:

		const PlanetHolder* const holder;
	};

}
}
#endif /* __GFX_PLANET_FACTORY_H__ */

