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
			Clusterer(MEM::MISC::BufferGl<float3> *positions);
			virtual ~Clusterer();

			// @brief Oblicza klastry na podstawie znanych buforów - ilość klastrów jest obliczana na podstawie ilości planet.
			void kmeans();

			/// @brief Zwraca ilość klastrów po klasteryzacji
			size_t getCount() const;

			const MEM::MISC::BufferCu<float3> *getCenters() const;
			const MEM::MISC::BufferCu<float> *getMasses() const;
			const MEM::MISC::BufferCu<unsigned> *getAssignments() const;

		protected:
			/// @brief Określa początkowe położenia środków klastrów
			void initClusters();
			/// @brief Oblicz jedną iterację algorytmu
			/// @returns Bład klasteryzacji
			float compute();

			/// @brief Sortuje przypisania do klastrów
			void sortByCluster();
			void reduceMeans();
			float reduceErrors();

			MEM::MISC::ClusterHolder m_holder;
			MEM::MISC::BufferGl<float3> *m_pPositions;
			MEM::MISC::BufferCu<float> m_errors;
			MEM::MISC::BufferCu<unsigned> m_shuffle;
			MEM::MISC::BufferCu<unsigned> m_counts;
	};
}

