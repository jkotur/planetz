#ifndef _PHX_KERNELS_H_
#define _PHX_KERNELS_H_
#include <string>

//#define PHX_DEBUG

namespace PHX
{
#ifdef PHX_DEBUG
	/// @brief Wyliczenie prędkości dla wszystkich.
	/// @details Zwykłe obliczenie zależności każdy-z-każdym. Używany, gdy klasteryzacja jest wyłączona.
	__global__ void basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt, float3 *dvs, unsigned who );
	
	/// @brief Wyliczenie prędkości w klastrze.
	/// @details Obliczenie prędkości planet na podstawie ich oddziaływań z innymi planetami w obrębie klastra.
	/// @param cluster Klaster, dla którego odbywają się obliczenia.
	__global__ void inside_cluster_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *shuffle, unsigned *counts, unsigned cluster, float3 *dvs, unsigned who, unsigned *whois );
#else
	/// @brief Wersja debugowa funkcji
	__global__ void basic_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *cnt );
	
	/// @brief Wersja debugowa funkcji
	__global__ void inside_cluster_interaction( float3 *positions, float *masses, float3 *velocities, unsigned *shuffle, unsigned *counts, unsigned cluster );
#endif

	/// @brief Oblicza prędkości całych klastrów
	__global__ void outside_cluster_interaction( float3 *centers, float *masses, unsigned count, float3 *velocities_impact );

	/// @brief Propaguje dV klastrów do planet w tych klastrach
	__global__ void propagate_velocities( float3 *velocities_impact, float3 *positions, float3 *velocities, unsigned *shuffle, unsigned *count, unsigned last_cluster );

	/// @brief Oblicza nowe pozycje na podstawie prędkości
	__global__ void update_positions_kernel( float3 *positions, float3 *velocities, unsigned *count );

	/// @brief Sprawdza, które planety kolidują ze sobą.
	/// @details Dla każdej planety, znajduje pierwszą, która z nią koliduje. Jej id jest zapisane w tablicy merges. Jeżeli planeta o id i nie koliduje z żadną inną, merges[i] = i.
	__global__ void detect_collisions( float3 *positions, float *radiuses, unsigned *count, unsigned *shuffle, unsigned last_cluster, unsigned *merges, unsigned *merge_needed );

	/// @brief Skleja planety wg tablicy in_merges. Jeżeli nie sklei wszystkiego, *done jest ustawiane na 0, a w out_merges są dane do kolejnego wywołania.
	__global__ void merge_collisions( unsigned *in_merges, unsigned *out_merges, float3 *positions, float3 *velocities, float *masses, float *radiuses, unsigned *count, unsigned *done );

	/// @brief Tworzy filtr do usuwania planet.
	/// @details Usunięte zostaną planety, których masa wynosi 0.
	__global__ void create_filter( float *masses, unsigned *filter, unsigned *count );

	/// @brief Pobiera tekst błędu.
	/// @details Jeżeli podczas działania kernela nie został spełniony warunek w CUDA_ASSERT, to zwrócona zostanie jego treść. W przeciwnym wypadku, zwrócony zostanie pusty ciąg znaków.
	std::string getErr();

}

#endif // _PHX_KERNELS_H_
