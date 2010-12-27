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
	struct PlanetHolderBase
	{
		PlanetHolderBase( unsigned num = 0 )
			: model(0)
			, emissive(0)
			, pos(0)
			, radius(0)
			, count(1)
			, mass(0)
			, velocity(0)
			, m_size(num)
		{
			resize( num );
		}

		virtual ~PlanetHolderBase()
		{
			log_printf(INFO, "deleted planetholder\n");
		}

		void resize(const size_t num)
		{
			TODO("keep previous data...");
			model     .resize(num);
			emissive  .resize(num);
			texId     .resize(num);
			atmosphere.resize(num);
			pos       .resize(num);
			radius    .resize(num);
			mass      .resize(num);
			velocity  .resize(num);

			count.assign( num );
			m_size = num;
		}

		//   * GFX
		GBUF<int>    model;
		GBUF<float>  emissive; // redundant to model, but needed for speed
		GBUF<int>    texId;
		GBUF<float>  atmosphere;

		//   * COMMON
		GBUF<float3>   pos;
		GBUF<float>    radius;
		GBUF<uint32_t> count;

		//   * PHX
		CBUF<float>    mass;
		CBUF<float3>   velocity;

		size_t size()
		{
			return m_size;
		}
	private:
		size_t m_size;
	};

	//template<template<class T>class CBUF, template<class S>class GBUF>
	//PlanetHolderBase<CBUF, GBUF>::PlanetHolderBase( unsigned num )

	//template<template<class T>class CBUF, template<class S>class GBUF>
	//PlanetHolderBase<CBUF, GBUF>::~PlanetHolderBase()

	//template<template<class T>class CBUF, template<class S>class GBUF>
	//void PlanetHolderBase<CBUF, GBUF>::resize(const size_t num)

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

	struct PointsCloudHolder
	{
		BufferCu<float>    phx_dt;

		BufferGl<float3>   pointsCloud_points;
		BufferGl<uint32_t> pointsCloud_size;
	};
}
}
#endif // HOLDER_H
