#include "phx.h"

using namespace CPU;

class Phx::CImpl
{
	public:
		CImpl(GPU::PlanetHolder *h);
		virtual ~CImpl();

		void compute(unsigned n);

	private:
		void map_buffers();
		void unmap_buffers();

		GPU::PlanetHolder *holder;
};

Phx::CImpl::CImpl(GPU::PlanetHolder *h)
	: holder(h)
{
}

Phx::CImpl::~CImpl()
{
}

void Phx::CImpl::compute(unsigned n)
{
	map_buffers();
	//run_clusters();
	//for(unsigned i = 0; i < n; ++i)
	//{
	//	run_nbodies();
	//}
	unmap_buffers();
}

void Phx::CImpl::map_buffers()
{
	holder->pos.map( GPU::BUF_CU );
	holder->radius.map( GPU::BUF_CU );
	holder->velocity.bind();
	holder->mass.bind();
	holder->count.map( GPU::BUF_CU );
}

void Phx::CImpl::unmap_buffers()
{
	holder->pos.unmap();
	holder->radius.unmap();
	holder->velocity.unbind();
	holder->mass.unbind();
	holder->count.unbind();
}

Phx::Phx(GPU::PlanetHolder *h)
	: impl( new CImpl(h) )
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
