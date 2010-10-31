#ifndef _CPU_PHX_H_
#define _CPU_PHX_H_

#include "../gpu/holder.h"

namespace CPU
{
	class Phx
	{
		public:
			Phx(GPU::PlanetHolder *h);
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

#endif // _CPU_PHX_H_

