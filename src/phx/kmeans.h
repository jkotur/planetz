#pragma once

#include <mem/misc/phx_planet_factory.h>
#include <cudpp.h>

const float EPSILON = 1e-5;

namespace PHX
{
	/// @brief Klasa odpowiedzialna za klasteryzację.
	/// @details Używa algorytmu k-means do podziału przestrzeni na klastry.
	class Clusterer
	{
		public:
			/// @brief Konstruktor 
			/// @param ppf wskaźnik na faktorię, z której klasa ma czerpać informacje.
			Clusterer( MEM::MISC::PhxPlanetFactory *ppf );
			virtual ~Clusterer();

			/// @brief Oblicza klastry na podstawie znanych buforów - ilość klastrów jest obliczana na podstawie ilości planet.
			void kmeans();

			/// @brief Zwraca ilość klastrów po klasteryzacji
			size_t getCount() const;

			/// @brief Zwraca środki klastrów
			MEM::MISC::BufferCu<float3> *getCenters();

			/// @brief Zwraca sumaryczne masy w klastrach
			MEM::MISC::BufferCu<float> *getMasses();

			/// @brief Zwraca mapowanie indeksów.
			/// @details Klastry określone są poprzez dwie tablice - shuffle oraz counts. Planeta shuffle[i] należy do klastra j <=> counts[j-1] <= i < counts[j]. counts[-1] jest umownie równe 0.
			MEM::MISC::BufferCu<unsigned> *getShuffle();

			/// @brief Zwraca liczności klastrów.
			/// @details Klastry określone są poprzez dwie tablice - shuffle oraz counts. Planeta shuffle[i] należy do klastra j <=> counts[j-1] <= i < counts[j]. counts[-1] jest umownie równe 0.
			MEM::MISC::BufferCu<unsigned> *getCounts();

		private:
			/// @brief Określa początkowe położenia środków klastrów
			void initClusters();

			/// @brief Oblicza jedną iterację algorytmu
			/// @returns Bład klasteryzacji
			float compute();

			/// @brief Sortuje przypisania do klastrów
			void sortByCluster();

			/// @brief Oblicza nowe środki klastrów
			void reduceMeans();

			/// @brief Oblicza błąd klasteryzacji
			float reduceErrors();

			/// @brief Oblicza parametry klastrów - masy
			/// @todo Być może promienie klastrów?
			void calcAttributes();

			/**
			 * @brief Inicjalizuje CUDPPHandle do sortowania.
			 *
			 * @param uint true, jeżeli inicjalizacja jest dla pętli
			 * głównej (klucz - unsigned int, rosnąco), natomiast 
			 * false - dla initClusters (klucz - float, malejąco).
			 */
			void initCudpp( bool uint=true );

			/**
			 * @brief Cleanup po initCudpp.
			 */
			void termCudpp();

			/**
			 * @brief Wybiera środki klastrów na podstawie mas planet.
			 */
			void massSelect();

			MEM::MISC::PhxPlanetFactory *m_planets;

			MEM::MISC::ClusterHolder m_holder;
			MEM::MISC::BufferCu<float> m_errors;
			MEM::MISC::BufferCu<unsigned> m_shuffle;
			MEM::MISC::BufferCu<unsigned> m_counts;

			unsigned m_prevSize;

			CUDPPHandle sortplan;
	};
}

