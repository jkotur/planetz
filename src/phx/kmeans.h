#pragma once

#include "mem/misc/holder.h"
#include "mem/misc/buffer.h"
#include "mem/misc/buffer_cu.hpp"

const float EPSILON = 1e-5;

namespace PHX
{
	class Clusterer
	{
		public:
			Clusterer(MEM::MISC::BufferGl<float3> *positions, MEM::MISC::BufferCu<float> *masses);
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
			/// @detail Klastry określone są poprzez dwie tablice - shuffle oraz counts. Planeta shuffle[i] należy do klastra j <=> counts[j-1] <= i < counts[j]. counts[-1] jest umownie równe 0.
			MEM::MISC::BufferCu<unsigned> *getShuffle();

			/// @brief Zwraca liczności klastrów.
			/// @detail Klastry określone są poprzez dwie tablice - shuffle oraz counts. Planeta shuffle[i] należy do klastra j <=> counts[j-1] <= i < counts[j]. counts[-1] jest umownie równe 0.
			MEM::MISC::BufferCu<unsigned> *getCounts();

		protected:
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

			MEM::MISC::BufferGl<float3> *m_pPositions;
			MEM::MISC::BufferCu<float> *m_pPlanetMasses;

			MEM::MISC::ClusterHolder m_holder;
			MEM::MISC::BufferCu<float> m_errors;
			MEM::MISC::BufferCu<unsigned> m_shuffle;
			MEM::MISC::BufferCu<unsigned> m_counts;

			unsigned m_prevSize;
	};
}

