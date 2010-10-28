
#ifndef __GFX_PLANET_FACTORY_H__

#define __GFX_PLANET_FACTORY_H__

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

		const GfxPlanet* getPlanet( int id ) const;

		BufferGl<uint8_t> *getModels() const;

		BufferGl<float3>  *getPositions() const;
		BufferGl<float>   *getRadiuses() const;
		BufferGl<uint32_t>*getCounts() const;

	private:

		const Holder* holder;
	};

}

#endif /* __GFX_PLANET_FACTORY_H__ */

