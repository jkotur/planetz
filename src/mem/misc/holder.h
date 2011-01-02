#ifndef HOLDER_H
#define HOLDER_H

#include "buffer.h"
#include "buffer_cpu.h"
#include "buffer_cu.hpp"

namespace MEM
{
namespace MISC
{
	/**
	 * @brief Szablon klasy agregującej bufory z informacjami o planetach.
	 */
	template<template<class T>class CBUF, template<class S>class GBUF>
	class PlanetHolderBase
	{
		public:
			/**
			 * @brief Tworzy nowy obiekt.
			 *
			 * @param num Ilość planet.
			 */
			PlanetHolderBase( unsigned num = 0 );
			virtual ~PlanetHolderBase();

			/**
			 * @brief Zmienia rozmiar holdera.
			 *
			 * @param num Nowa ilość planet.
			 */
			void resize(const size_t num);

			/**
			 * @brief Pobiera rozmiar bufora.
			 *
			 * @returns Ilość planet w buforze.
			 */
			size_t size() const;

			/**
			 * @brief Usuwa planety na podstawie podanej maski.
			 * 
			 * @param mask Bufor zawierający 0 i 1. Planety o indeksach, pod którymi mask ma wartość 0 zostaną usunięte.
			 */
			void filter( BufferCu<unsigned> *mask );

			//   * GFX
			/**
			 * @brief Bufor zawierający id modeli planet.
			 */
			GBUF<int>    model;

			/**
			 * @brief ???
			 *
			 * @todo Kuba, udokumentuj mnie!
			 */
			GBUF<float>  emissive; // redundant to model, but needed for speed

			/**
			 * @brief ???
			 *
			 * @todo Kuba, udokumentuj mnie!
			 */
			GBUF<int>    texId;

			/**
			 * @brief ???
			 *
			 * @todo Kuba, udokumentuj mnie!
			 */
			GBUF<float3> atm_color;

			/**
			 * @brief ???
			 *
			 * @todo Kuba, udokumentuj mnie!
			 */
			GBUF<float2> atm_data;

			//   * COMMON
			/**
			 * @brief Bufor z pozycjami planet.
			 */
			GBUF<float3>   pos;

			/**
			 * @brief Bufor z promieniami planet.
			 */
			GBUF<float>    radius;

			/**
			 * @brief Bufor jednoelementowy zawierający ilość planet.
			 *
			 * @todo Sprawdzić, czy to jest jeszcze potrzebne.
			 */
			GBUF<uint32_t> count;

			//   * PHX
			/**
			 * @brief Bufor z masami planet.
			 */
			CBUF<float>    mass;

			/**
			 * @brief Bufor z wektorami prędkości planet.
			 */
			CBUF<float3>   velocity;

		private:
			size_t m_size;
			size_t m_realsize;
	};

	/**
	 * @brief Kontener buforów dla GPU - używany w fizyce.
	 */
	typedef PlanetHolderBase< BufferCu, BufferGl > PlanetHolder;
	/**
	 * @brief Kontener buforów dla CPU - używany przy zapisie do pliku.
	 */
	typedef PlanetHolderBase< BufferCpu, BufferCpu > CpuPlanetHolder;

	/**
	 * @brief Klasa agregująca bufory z informacjami o klastrach.
	 */
	class ClusterHolder // taki tam class - w sumie zjebana enkapsulacja w tych holderach, skoro każdy i tak trzyma buffer jako public membera
	{
		public:
			ClusterHolder();
			virtual ~ClusterHolder();

			/**
			 * @brief Środki klastrów.
			 */
			BufferCu<float3>    centers;

			/**
			 * @brief Sumaryczne masy w klastrze
			 */
			BufferCu<float>     masses;

			/**
			 * @brief Przyporządkowania planet do klastrów.
			 *
			 * @details Planeta i należy do klastra j <=> assignments[i] = j
			 */
			BufferCu<unsigned>  assignments;

			/**
			 * @brief Zmienia rozmiar buforów.
			 *
			 * @param k_size Nowa ilość klastrów.
			 *
			 * @param n_size Nowa ilość planet.
			 */
			void resize(size_t k_size, size_t n_size);

			/**
			 * @brief Zwraca ilość klastrów przechowywaną w ClusterHolderze.
			 */
			size_t k_size() const;

		private:
			size_t m_size;
	};

	template<template<class T>class CBUF, template<class S>class GBUF>
	PlanetHolderBase<CBUF, GBUF>::PlanetHolderBase( unsigned num )
		: model(0)
		, emissive(0)
		, pos(0)
		, radius(0)
		, count(1)
		, mass(0)
		, velocity(0)
		, m_size(0) // ustawi się w resize
		, m_realsize(0) // jw.
	{
		resize( num );
	}

	template<template<class T>class CBUF, template<class S>class GBUF>
	PlanetHolderBase<CBUF, GBUF>::~PlanetHolderBase()
	{
		log_printf(INFO, "deleted planetholder\n");
	}

	template<template<class T>class CBUF, template<class S>class GBUF>
	void PlanetHolderBase<CBUF, GBUF>::resize(const size_t num)
	{
		if( num > m_realsize ) // ew. można zmniejszać kiedy num << m_realsize
		{
			TODO("keep previous data...");
			model    .resize(num);
			emissive .resize(num);
			texId    .resize(num);
			atm_color.resize(num);
			atm_data .resize(num);
			pos      .resize(num);
			radius   .resize(num);
			mass     .resize(num);
			velocity .resize(num);
			m_realsize = num;
		}
		count.assign( num );
		m_size = num;
	}

	template<template<class T>class CBUF, template<class S>class GBUF>
	size_t PlanetHolderBase<CBUF, GBUF>::size() const
	{
		return m_size;
	}

	/**
	 * @brief Wzorzec klasy pomocniczej, służacej do filtrowania PlanetHolderBase'a.
	 */
	template<template<class T>class CBUF, template<class S>class GBUF>
	class PlanetHolderFilterFunctor
	{
		public:
			/**
			 * @brief Tworzy funktor.
			 *
			 * @param owner Wskaźnik na klasę, która ma być filtrowana.
			 */
			PlanetHolderFilterFunctor( PlanetHolderBase<CBUF, GBUF> *owner );

			/**
			 * @brief Filtruje zadanego holdera.
			 *
			 * @param mask Maska filtrowania.
			 */
			void operator()( BufferCu<unsigned> *mask );
	};

	void __filter( PlanetHolder *what, BufferCu<unsigned> *how );

	/**
	 * @brief Specjalizacja, używana przez PlanetHolder'a.
	 */
	template<>
	class PlanetHolderFilterFunctor< BufferCu, BufferGl >
	{
		public:
			/**
			 * @brief Tworzy funktor.
			 *
			 * @param _owner Wskaźnik na klasę, która ma być filtrowana.
			 */
			PlanetHolderFilterFunctor( PlanetHolder *_owner )
				: owner( _owner )
			{}

			/**
			 * @brief Filtruje zadanego holdera.
			 *
			 * @param mask Maska filtrowania.
			 */
			void operator()( BufferCu<unsigned> *mask )
			{
				__filter( owner, mask );
			}
		private:
			PlanetHolder *owner;
	};

	template<template<class T>class CBUF, template<class S>class GBUF>
	void PlanetHolderBase<CBUF, GBUF>::filter( BufferCu<unsigned> *mask )
	{
		PlanetHolderFilterFunctor<CBUF, GBUF> filt( this );
		filt( mask );
		m_size = count.retrieve();
	}
}
}
#endif // HOLDER_H
