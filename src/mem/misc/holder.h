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
			model.resize(num);
			pos.resize(num);
			radius.resize(num);
			mass.resize(num);
			velocity.resize(num);

			count.assign( num );
			m_size = num;
		}

		//   * GFX
		GBUF<int>    model;

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

	struct PointsCloudHolder
	{
		BufferCu<float>    phx_dt;

		BufferGl<float3>   pointsCloud_points;
		BufferGl<uint32_t> pointsCloud_size;
	};
}
}
#endif // HOLDER_H
