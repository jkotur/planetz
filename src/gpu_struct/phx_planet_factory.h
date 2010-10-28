
#ifndef __PHX_PLANET_FACTORY_H__

#define __PHX_PLANET_FACTORY_H__

namespace GPU {

	class PhxPlanet {
	public:
		PhxPlanet( int id );
		virtual ~PhxPlanet();
		
	private:
		
	};


	class PhxPlanetFactory {
	public:
		PhxPlanetFactory( const Holder* );
		virtual ~PhxPlanetFactory( );

		PhxPlanet* getPlanet( int id );

		BufferGl<uint8_t> *getModels();

		BufferGl<float3>  *getPositions();
		BufferGl<float>   *getRadiuses();
		BufferGl<uint32_t>*getCounts();

	private:

		const Holder* holder;
	};

}

#endif /* __PHX_PLANET_FACTORY_H__ */

