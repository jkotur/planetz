#include "phx.h"
#include "phx_kernels.h"
#include "kmeans.h"

using namespace PHX;

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
	run_clusters();
	for(unsigned i = 0; i < n; ++i)
	{
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
	unsigned mem = block.x * ( sizeof(float3) + sizeof(float) );

	basic_interaction<<<grid, block, mem>>>( 
		planets->getPositions().map(MEM::MISC::BUF_CU), 
		planets->getMasses().d_data(), 
		planets->getVelocities().d_data(),
		planets->getCount().map(MEM::MISC::BUF_CU) );
	CUT_CHECK_ERROR("Kernel launch");
}

void Phx::CImpl::run_clusters()
{
	clusterer.kmeans();
	MEM::MISC::BufferCu<unsigned> *count = const_cast<MEM::MISC::BufferCu<unsigned>*>(clusterer.getCounts());
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

