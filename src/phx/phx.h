#ifndef _PHX_PHX_H_
#define _PHX_PHX_H_

#include <mem/misc/phx_planet_factory.h>

/// @brief Przestrzeń nazw dla obiektów odpowiedzialnych za fizykę.
namespace PHX
{
	/// @brief Główna klasa odpowiedzialna za obliczenia fizyczne.
	class Phx
	{
		public:
			/// @brief Inicjalizacja fizyki.
			Phx(MEM::MISC::PhxPlanetFactory *p);
			virtual ~Phx();

			/// @brief Oblicza n klatek fizyki
			/// @param n Ilość klatek
			virtual void compute(unsigned n=1);

			/// @brief Włącza/wyłącza klasteryzację.
			virtual void enableClusters(bool orly=true);

			/// @brief Sprawdza, czy klasteryzacja jest włączona.
			virtual bool clustersEnabled() const;

		private:
			class CImpl;
			CImpl* impl;
	};
}

#endif // _PHX_PHX_H_

