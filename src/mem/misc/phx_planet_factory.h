#ifndef __PHX_PLANET_FACTORY_H__

#define __PHX_PLANET_FACTORY_H__

#include "buffer.h"
#include "holder.h"

namespace MEM
{
namespace MISC
{
	/**
	 * @brief Klasa określająca planetę z punktu widzenia fizyki.
	 * Pozwala odczytywać z CPU fizyczne własności planety.
	 */
	class PhxPlanet
	{
	public:
		PhxPlanet();

		/**
		 * @brief Tworzy obiekt planety o zadanym id.
		 * Konstruktor powinien być używany jedynie przez PhxPlanetFactory.
		 */
		PhxPlanet( unsigned id , PlanetHolder* h );
		PhxPlanet( const PhxPlanet& );
		virtual ~PhxPlanet();

		PhxPlanet& operator=( const PhxPlanet& rhs );

		int getId() const;
		
		/**
		 * @brief Pozycja planety.
		 */
		float3  getPosition() const;

		/**
		 * @brief Promień planety.
		 */
		float   getRadius() const;

		/**
		 * @brief Masa planety.
		 */
		float	getMass() const;

		/**
		 * @brief Prędkość planety.
		 */
		float3	getVelocity() const;

		/**
		 * @brief Sprawdza, czy obiekt zawiera poprawne dane.
		 */
		bool isValid() const;

		/**
		 * @brief Usuwa planetę z wszechświata.
		 */
		void remove();

	private:
		void initFromOther( const PhxPlanet& );
		PlanetLogin login;
		PlanetHolder* holder;
		bool exists;
	};

	/**
	 * @brief Klasa będąca enkapsulacją PlanetHoldera dla obiektów fizyki.
	 * Zawiera tylko te bufory, które są niezbędne do obliczeń fizycznych.
	 * Można pobierać z niej całe bufory jednego rodzaju danych (np. pozycje)
	 * lub pobrać obiekt PhxPlanet.
	 */
	class PhxPlanetFactory
	{
	public:
		/**
		 * @brief Konstruktor.
		 *
		 * @param holder Źródło danych.
		 */
		PhxPlanetFactory( PlanetHolder* holder );
		virtual ~PhxPlanetFactory( );

		/**
		 * @brief Tworzy i zwraca PhxPlanet - interfejs do pobierania informacji o konkretnej planecie.
		 */
		PhxPlanet getPlanet( int id );

		/**
		 * @brief Bufor z pozycjami planet.
		 */
		BufferGl<float3>  &getPositions();

		/**
		 * @brief Bufor z promieniami planet.
		 */
		BufferGl<float>   &getRadiuses();

		/**
		 * @brief Bufor z masami planet.
		 */
		BufferCu<float>   &getMasses();

		/**
		 * @brief Bufor z prędkościami planet.
		 */
		BufferCu<float3>  &getVelocities();

		/**
		 * @brief Bufor z ilością planet.
		 */
		BufferGl<uint32_t>&getCount();

		/**
		 * @brief Ilość planet.
		 */
		unsigned size() const;

		/**
		 * @brief Filtrowanie planet - usuwa te planety, które mają wartość 0 w mask.
		 */
		void filter( BufferCu<unsigned> *mask );

	private:

		PlanetHolder* holder;
	};

}
}
#endif /* __PHX_PLANET_FACTORY_H__ */

