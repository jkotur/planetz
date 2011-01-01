#ifndef HOLDER_H
#define HOLDER_H

#include "buffer.h"
#include "buffer_cpu.h"
#include "buffer_cu.hpp"

namespace MEM
{
namespace MISC
{
	template<template<class T>class CBUF, template<class S>class GBUF>
	class PlanetHolderBase
	{
		public:
			PlanetHolderBase( unsigned num = 0 );
			virtual ~PlanetHolderBase();

			void resize(const size_t num);
			size_t size() const;

			void filter(BufferCu<unsigned> *deleted);

			//   * GFX
			GBUF<int>    model;
			GBUF<float>  emissive; // redundant to model, but needed for speed
			GBUF<int>    texId;
			GBUF<float3> atm_color;
			GBUF<float2> atm_data;

			//   * COMMON
			GBUF<float3>   pos;
			GBUF<float>    radius;
			GBUF<uint32_t> count;

			//   * PHX
			CBUF<float>    mass;
			CBUF<float3>   velocity;

		private:
			size_t m_size;
			size_t m_realsize;
	};

	typedef PlanetHolderBase< BufferCu, BufferGl > PlanetHolder;
	typedef PlanetHolderBase< BufferCpu, BufferCpu > CpuPlanetHolder;

	class ClusterHolder // taki tam class - w sumie zjebana enkapsulacja w tych holderach, skoro każdy i tak trzyma buffer jako public membera
	{
		public:
			ClusterHolder();
			virtual ~ClusterHolder();

			// środki klastrów
			BufferCu<float3>    centers;

			// sumaryczne masy w klastrze
			BufferCu<float>     masses;

			// przyporządkowania planet do klastrów
			// planeta i należy do klastra j <=> assignments[i] = j
			BufferCu<unsigned>  assignments;

			void resize(size_t k_size, size_t n_size);
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

	template<template<class T>class CBUF, template<class S>class GBUF>
	class PlanetHolderFilterFunctor
	{
		public:
			PlanetHolderFilterFunctor( PlanetHolderBase<CBUF, GBUF> *owner );
			void operator()( BufferCu<unsigned> *mask );
	};

	void __filter( PlanetHolder *what, BufferCu<unsigned> *how );

	template<>
	class PlanetHolderFilterFunctor< BufferCu, BufferGl >
	{
		public:
			PlanetHolderFilterFunctor( PlanetHolder *_owner )
				: owner( _owner )
			{}

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
