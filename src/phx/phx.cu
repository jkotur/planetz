#include "phx.h"
#include "phx_kernels.h"

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

		MEM::MISC::PhxPlanetFactory *planets;
};

Phx::CImpl::CImpl(MEM::MISC::PhxPlanetFactory *p)
	: planets(p)
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
	//run_clusters();
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

	basic_interaction<<<grid, block>>>( 
		planets->getPositions().map(MEM::MISC::BUF_CU), 
		planets->getMasses().d_data(), 
		planets->getVelocities().d_data(),
		planets->getCount().map(MEM::MISC::BUF_CU) );
	CUT_CHECK_ERROR("Kernel launch");
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

