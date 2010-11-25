#ifndef _PHX_PHX_H_
#define _PHX_PHX_H_

#include <mem/misc/phx_planet_factory.h>

namespace PHX
{
	class Phx
	{
		public:
			Phx(MEM::MISC::PhxPlanetFactory *p);
			virtual ~Phx();

			/// @brief Obliczenie n klatek fizyki
			/// @param n Ilość klatek
			/// @details Przed właściwym obliczeniem następuje klasteryzacja planet. W przypadku n = 0 metoda compute podzieli przestrzeń na klastry, nie przemieszczając samych planet.
			virtual void compute(unsigned n=1);

		private:
			class CImpl;
			CImpl* impl;
	};
}

#endif // _PHX_PHX_H_

