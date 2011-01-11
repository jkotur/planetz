#ifndef __HOLDER_CLEANER_H__
#define __HOLDER_CLEANER_H__

#include "phx_planet_factory.h"
#include "util/event.h"

namespace MEM
{
namespace MISC
{
/**
 * @brief Klasa czyszcząca PlanetHoldera, gdy ilość nieużywanych (skasowanych) planet jest zbyt duża.
 */
class PlanetHolderCleaner
{
	public:
		/**
		 * @brief Przypisuje pamięć, która ma być czyszczona.
		 *
		 * @param h PhxPlanetFactory reprezentujący odpowiedniego PlanetHoldera.
		 */
		PlanetHolderCleaner( PhxPlanetFactory *h );
		virtual ~PlanetHolderCleaner();

		/**
		 * @brief Informuje cleanera, że należy przy najbliższej okazji sprawdzić, czy nie jest konieczny cleanup.
		 */
		void notifyCheckNeeded();

		/**
		 * @brief Sprawdza, czy należy odfiltrować PlanetHoldera i robi to, kiedy należy.
		 */
		void work();
	
		/**
		 * @brief Zapisuje się na zmianę id planety.
		 *
		 * @param callback Metoda wołana w reakcji na zmianę id.
		 *
		 * @param id Monitorowane ID.
		 */
	//	void subscribeForIdChange( EventDelegate1<unsigned> callback, unsigned id );

		/**
		 * @brief Wypisuje się z informowania o zmianie planety.
		 */
		//void unsubscribeForIdChange( EventDelegate1<unsigned> callback );

	private:
		void createFilter();
		bool filteringNeeded();
		void filterHolder();

		PhxPlanetFactory *fact;
		MEM::MISC::BufferCu<unsigned> filter;
		MEM::MISC::BufferCu<unsigned> planetsInUse;
		bool needChecking;
};
} // MISC
} // MEM

#endif // __HOLDER_CLEANER_H__
