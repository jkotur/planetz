#ifndef HOLDER_H
#define HOLDER_H

#include "buffer.h"
#include "buffer_cpu.h"
#include "buffer_cu.hpp"
#include <map>

namespace MEM
{
namespace MISC
{
	typedef unsigned int PlanetLogin;
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
			 * @param mask Bufor zawierający 0 i 1. Planety o
			 * indeksach, pod którymi mask ma wartość 0 zostaną
			 * usunięte.
			 */
			void filter( BufferCu<unsigned> *mask );

			PlanetLogin createLogin( unsigned id );
			void releaseLogin( PlanetLogin pl );
			int actualID( PlanetLogin pl );

			//
			//   * GFX
			//
			/** @brief Bufor zawierający id modeli planet.  */
			GBUF<int>    model;

			/** @brief Bufor z informacją kolorze planety */
			GBUF<float4> color;

			/** @brief Bufor z informacją o reakcji na światło */
			GBUF<float3> light;

			/** @brief Bufor z informacją o numerze tekstury */
			GBUF<int>    texId;

			/** @brief Bufor z informacją o kolorze atmosfery */
			GBUF<float3> atm_color;

			/** @brief Bufor z dodatkowymi danymi o atmosferze (promień,gęstość) */
			GBUF<float2> atm_data;

			//
			//   * COMMON
			//
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

			//
			//   * PHX
			//
			/**
			 * @brief Bufor z masami planet.
			 */
			CBUF<float>    mass;

			/**
			 * @brief Bufor z wektorami prędkości planet.
			 */
			CBUF<float3>   velocity;

		private:
			/**
			 * @brief Rozmiar buforów w użyciu.
			 */
			size_t m_size;
			/**
			 * @brief Fizyczny, zaalokowany rozmiar buforów.
			 */
			size_t m_realsize;

			typedef std::map<PlanetLogin, int> IdMap;

			IdMap m_planetIds;
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
		, light(0)
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
		ASSERT_MSG( m_planetIds.empty(), "There are still planets in use!" );
		log_printf(INFO, "deleted planetholder\n");
	}

	template<template<class T>class CBUF, template<class S>class GBUF>
	void PlanetHolderBase<CBUF, GBUF>::resize(const size_t num)
	{
		if( num > m_realsize ) // ew. można zmniejszać kiedy num << m_realsize
		{
			TODO("keep previous data...");
			model    .resize(num);
			light    .resize(num);
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

	typedef std::map<const unsigned, int> IdxChangeSet;

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
			 *
			 * @param changes Zbiór śledzonych indeksów.
			 *
			 * @returns Nowy rozmiar.
			 */
			unsigned operator()( BufferCu<unsigned> *mask, IdxChangeSet *changes );
	};

	unsigned __filter( PlanetHolder *what, BufferCu<unsigned> *how, IdxChangeSet *changes );

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
			 *
			 * @param changes Zbiór śledzonych indeksów.
			 *
			 * @returns Nowy rozmiar.
			 */
			unsigned operator()( BufferCu<unsigned> *mask, IdxChangeSet *changes )
			{
				return __filter( owner, mask, changes );
			}
		private:
			PlanetHolder *owner;
	};

	template<template<class T>class CBUF, template<class S>class GBUF>
	void PlanetHolderBase<CBUF, GBUF>::filter( BufferCu<unsigned> *mask )
	{
		PlanetHolderFilterFunctor<CBUF, GBUF> filt( this );
		IdxChangeSet changes;
		
		for( IdMap::iterator it = m_planetIds.begin(); it != m_planetIds.end(); ++it )
		{
			if( it->second >= 0 )
			{
				const unsigned id = it->second;
				changes.insert( std::make_pair( id, -1 ) );
			}
		}
		m_size = filt( mask, &changes );
		IdxChangeSet::iterator ch_it = changes.begin();
		for( IdMap::iterator it = m_planetIds.begin(); it != m_planetIds.end(); ++it )
		{
			if( ch_it == changes.end() ) return;
			if( (int)ch_it->first == it->second )
			{
				it->second = ch_it->second;
			}
		}
	}

	template<template<class T>class CBUF, template<class S>class GBUF>
	PlanetLogin
	PlanetHolderBase<CBUF, GBUF>::createLogin( unsigned id )
	{
		ASSERT( id < m_size );
		if( m_planetIds.empty() )
		{
			m_planetIds[0] = id;
			return 0;
		}
		PlanetLogin login = m_planetIds.rbegin()->first + 1;
		while( m_planetIds.find( login ) != m_planetIds.end() ) ++login;
		m_planetIds[ login ] = id;
		return login;
	}

	template<template<class T>class CBUF, template<class S>class GBUF>
	void PlanetHolderBase<CBUF, GBUF>::releaseLogin( PlanetLogin login )
	{
		IdMap::iterator it = m_planetIds.find( login );
		ASSERT( it != m_planetIds.end() );
		m_planetIds.erase( it );	
	}

	template<template<class T>class CBUF, template<class S>class GBUF>
	int PlanetHolderBase<CBUF, GBUF>::actualID( PlanetLogin login )
	{
		IdMap::iterator it = m_planetIds.find( login );
		ASSERT( it != m_planetIds.end() );
		return it->second;
	}
}
}
#endif // HOLDER_H
