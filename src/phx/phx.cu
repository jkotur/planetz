#include <limits>
#include "phx.h"
#include "phx_kernels.h"
#include "kmeans.h"

using namespace PHX;

inline bool operator==(const float3& l, const float3& r)
{
	return 0 == memcmp( &l, &r, sizeof(float3) );
}

template<class T>
class Validator
{
	public:
		void validate( const T& value )
		{
		}
};

template<>
class Validator<float>
{
	public:
		void validate( const float& value )
		{
			//log_printf( DBG, "%f\n", value );
			ASSERT( !isnan(value) );
		}
};

template<>
class Validator<float3>
{
	public:
		void validate( const float3& value )
		{
			//log_printf( DBG, "[%f, %f, %f]\n", value.x, value.y, value.z );
			ASSERT( !isnan(value.x) );
			ASSERT( !isnan(value.y) );
			ASSERT( !isnan(value.z) );
		}
};

template<class T, template<class S> class BUF>
class BufferAdapter
{
	public:
		BufferAdapter( BUF<T>& );

		T* hostData();
};

template<class T>
class BufferAdapter<T, MEM::MISC::BufferGl >
{
	public:
		BufferAdapter( MEM::MISC::BufferGl<T>& b ):buf(b)
		{
		}

		T* hostData()
		{
			return buf.map( MEM::MISC::BUF_H );
		}
	private:
		MEM::MISC::BufferGl<T> &buf;
};

template<class T>
class BufferAdapter<T, MEM::MISC::BufferCu>
{
	public:
		BufferAdapter( MEM::MISC::BufferCu<T>& b ):buf(b)
		{
			buf.bind();
		}
		T *hostData()
		{
			return buf.h_data();
		}
		~BufferAdapter()
		{
			buf.unbind();
		}
	private:
		MEM::MISC::BufferCu<T> &buf;
};

template<class T, template<class S> class BUF>
class ConstChecker
{
	public:
		ConstChecker() : buf(NULL), data(NULL) {}
		virtual ~ConstChecker(){if(data)delete[]data;}
		void setBuf(BUF<T> *b)
		{
			buf = b;
			size = b->getLen();
			if( data )
			{
				delete []data;
			}
			data = new T[ size ];
			BufferAdapter<T, BUF> ad( *buf );
			ASSERT( ad.hostData() );
			memcpy( data, ad.hostData(), b->getSize() );
			Validator<T> v;
			for( unsigned i = 0; i < size; ++i )
			{
				//log_printf( DBG, "data[%u] ", i );
				v.validate( data[i] );
			}
		}

		void checkBuf()
		{
			T *actual_data = new T[ size ];
			ASSERT( actual_data != NULL );
			BufferAdapter<T, BUF> ad( *buf );
			memcpy( actual_data, ad.hostData(), buf->getSize() );
			Validator<T> v;
			for( unsigned i = 0; i < size; ++i )
			{
				//log_printf( DBG, "actual_data[%u] ", i );
				v.validate( actual_data[i] );
				ASSERT( actual_data[i] == data[i] );
			}
			delete []actual_data;
		}
	private:
		BUF<T> *buf;
		unsigned size;
		T *data;
};

ConstChecker<float3, MEM::MISC::BufferGl> pos_checker;
ConstChecker<float, MEM::MISC::BufferCu> mass_checker;
ConstChecker<float3, MEM::MISC::BufferCu> vel_checker;

class Phx::CImpl
{
	public:
		CImpl(MEM::MISC::PhxPlanetFactory *p);
		virtual ~CImpl();

		void compute(unsigned n);

	private:
		void map_buffers();
		void unmap_buffers();

		void run_nbodies( unsigned planet_count );
		void run_clusters();

		MEM::MISC::PhxPlanetFactory *planets;
		Clusterer clusterer;

		MEM::MISC::BufferCu<float3> tmp_pos;
		MEM::MISC::BufferCu<float3> tmp_vel;
		MEM::MISC::BufferCu<float> tmp_mass;
};

Phx::CImpl::CImpl(MEM::MISC::PhxPlanetFactory *p)
	: planets(p)
	, clusterer( &p->getPositions(), &p->getMasses() )
{
}

Phx::CImpl::~CImpl()
{
}

void Phx::CImpl::compute(unsigned n)
{
	unsigned planet_count;
	if( !(planet_count = planets->size()) )
		return;
	map_buffers();
	mass_checker.setBuf( &planets->getMasses() );
	run_clusters();
	mass_checker.checkBuf();
	
	for(unsigned i = 0; i < n; ++i)
	{
		vel_checker.setBuf( &planets->getVelocities() );
		vel_checker.checkBuf();
		run_nbodies( planet_count );
	}
	unmap_buffers();
}

void Phx::CImpl::map_buffers()
{
	planets->getPositions().map( MEM::MISC::BUF_CU );
	planets->getRadiuses().map( MEM::MISC::BUF_CU );
	planets->getCount().map( MEM::MISC::BUF_CU );
}

void Phx::CImpl::unmap_buffers()
{
	planets->getPositions().unmap();
	planets->getRadiuses().unmap();
	planets->getCount().unmap();
}

void Phx::CImpl::run_nbodies( unsigned threads )
{	
	ASSERT( threads );
	dim3 block( min( threads, 512 ) );
	dim3 grid( 1 + (threads - 1) / block.x );
	//unsigned mem = block.x * ( sizeof(float3) + sizeof(float) );
	tmp_pos.resize( threads );
	tmp_vel.resize( threads );
	//TODO( "dać te resize'y gdzieś indziej" );

#ifdef PHX_DEBUG
	float *d_shr, *d_glb;
	unsigned *d_idx;
	cudaMalloc( &d_shr, sizeof(float) );
	cudaMalloc( &d_glb, sizeof(float) );
	cudaMalloc( &d_idx, threads * sizeof(unsigned) );
	cudaMemset( d_idx, 0, threads * sizeof(unsigned) );
#endif
	pos_checker.setBuf( &planets->getPositions() );
	basic_interaction<<<grid, block>>>( 
		planets->getPositions().map(MEM::MISC::BUF_CU), 
		planets->getMasses().d_data(), 
		planets->getVelocities().d_data(),
		planets->getCount().map(MEM::MISC::BUF_CU),
		tmp_pos.d_data(),
		tmp_vel.d_data()
#ifdef PHX_DEBUG
		, d_shr, d_glb, d_idx
#endif
		);
	CUT_CHECK_ERROR("Kernel launch");
	
	pos_checker.checkBuf();
	
	cudaMemcpy( planets->getPositions().map(MEM::MISC::BUF_CU), tmp_pos.d_data(), threads * sizeof(float3), cudaMemcpyDeviceToDevice );
	cudaMemcpy( planets->getVelocities().d_data(), tmp_vel.d_data(), threads * sizeof(float3), cudaMemcpyDeviceToDevice );
#ifdef PHX_DEBUG
	float h_shr, h_glb;
	unsigned *h_idx = new unsigned[ threads ];
	cudaMemcpy( &h_shr, d_shr, sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( &h_glb, d_glb, sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_idx, d_idx, threads * sizeof(unsigned), cudaMemcpyDeviceToHost );
	std::string err = getErr();
	if( !err.empty() )
	{
		log_printf( _ERROR, "CUDA assertion failed: '%s'\n", err.c_str() );
		NOENTRY();
	}
	delete h_idx;
#endif
}

void Phx::CImpl::run_clusters()
{
	clusterer.kmeans();
}

Phx::Phx(MEM::MISC::PhxPlanetFactory *p)
	: impl( new CImpl(p) )
{
}

Phx::~Phx()
{
	delete impl;
}

void Phx::compute(unsigned n)
{
	impl->compute(n);
}

