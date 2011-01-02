
#ifndef __GFX_PLANET_FACTORY_H__

#define __GFX_PLANET_FACTORY_H__

#include "buffer.h"
#include "holder.h"

namespace MEM
{
namespace MISC
{
	/** 
	 * @brief Klasa będąca enkapsulacją holdera dla obiektów grafiki.
	 * Zawiera tylko te bufory które są niezbędne do wyświetlenia planet.
	 * Dodatkowo wszystkie bufory są stałe, dzięki czemu grafika nie może,
	 * a w zasadzie wie że nie powinna, modyfikować ich wartości.
	 */
	class GfxPlanetFactory {
	public:
		/** 
		 * @brief Ustawia klasę na podstawie holdera posiadającego
		 * wszystkie informacjie o planetach.
		 * 
		 * @param 
		 */
		GfxPlanetFactory( const PlanetHolder* );
		/** 
		 * @brief Usuwa klasę, nie czyszcząć żadnej pamięci w holderze.
		 */
		virtual ~GfxPlanetFactory( );

		/** 
		 * @brief Zwraca bufor z id modelu
		 * 
		 * @return bufor
		 */
		const BufferGl<int>     &getModels   () const;
		/** 
		 * @brief Zwraca bufor z informacją o emisji światła przez planetę
		 * 
		 * @return bufor
		 */
		const BufferGl<float>   &getEmissive () const;
		/** 
		 * @brief Zwraca bufor z informacją o numerze tekstury
		 * 
		 * @return bufor
		 */
		const BufferGl<int>     &getTexIds   () const;
		/** 
		 * @brief Zwraca bufor z informacją o kolorze atmosfery
		 * 
		 * @return bufor
		 */
		const BufferGl<float3>  &getAtmColor () const;
		/** 
		 * @brief Zwraca bufor z dodatkowymi danymi o atmosferze (promień,gęstość)
		 * 
		 * @return bufor
		 */
		const BufferGl<float2>  &getAtmData  () const;

		/** 
		 * @brief Zwraca bufor z informacją o pozycji planety w przestrzeni
		 * 
		 * @return bufor
		 */
		const BufferGl<float3>  &getPositions() const;
		/** 
		 * @brief Zwraca bufor z informacją o promieniu planety
		 * 
		 * @return bufor
		 */
		const BufferGl<float>   &getRadiuses () const;
		/** 
		 * @brief Zwraca bufor z informacją o ilości planet
		 * 
		 * @return bufor
		 */
		const BufferGl<uint32_t>&getCounts   () const;

		/**
		 * @brief Zwraca ilość planet.
		 */
		unsigned size() const;

	private:

		const PlanetHolder* const holder;
	};

}
}
#endif /* __GFX_PLANET_FACTORY_H__ */

