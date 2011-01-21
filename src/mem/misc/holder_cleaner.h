#ifndef __HOLDER_CLEANER_H__
#define __HOLDER_CLEANER_H__

#include "phx_planet_factory.h"

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
		 * @brief Polityka filtrowania przez PlanetHolderCleaner'a.
		 */
		enum FilteringPolicy
		{
			/**
			 * @brief Filtruje zawsze.
			 */
			Always = 0,
			/**
			 * @brief Filtruje często.
			 */
			Frequently,
			/**
			 * @brief Filtruje rzadko.
			 */
			Rarely,
			/**
			 * @brief Nie filtruje.
			 */
			Never
		};
		/**
		 * @brief Przypisuje pamięć, która ma być czyszczona.
		 *
		 * @param h PhxPlanetFactory reprezentujący odpowiedniego PlanetHoldera.
		 *
		 * @param p Ustawiona polityka filtrowania.
		 */
		PlanetHolderCleaner( PhxPlanetFactory *h, FilteringPolicy p = Rarely );
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
		 * @brief Wymusza filtrowanie PlanetHolder'a.
		 */
		void forceFilter();

		/**
		 * @brief Ustawia politykę filtrowania.
		 *
		 * @todo getFilteringPolicy? Potrzebne to komuś do czegoś?
		 */
		void setFilteringPolicy( FilteringPolicy p );

	private:
		void createFilter();
		bool filteringNeeded();
		void filterHolder();

		PhxPlanetFactory *fact;
		MEM::MISC::BufferCu<unsigned> filter;
		MEM::MISC::BufferCu<unsigned> planetsInUse;
		bool needChecking;
		FilteringPolicy filteringPolicy;
};
} // MISC
} // MEM

#endif // __HOLDER_CLEANER_H__
