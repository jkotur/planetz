#include "phx.h"

using namespace CPU;

class Phx::CImpl
{
	public:
		CImpl(MEM::MISC::PlanetHolder *h);
		virtual ~CImpl();

		void compute(unsigned n);

	private:
		void map_buffers();
		void unmap_buffers();

		MEM::MISC::PlanetHolder *holder;
};

Phx::CImpl::CImpl(MEM::MISC::PlanetHolder *h)
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
	holder->pos.map( MEM::MISC::BUF_CU );
	holder->radius.map( MEM::MISC::BUF_CU );
	holder->velocity.bind();
	holder->mass.bind();
	holder->count.map( MEM::MISC::BUF_CU );
}

void Phx::CImpl::unmap_buffers()
{
	holder->pos.unmap();
	holder->radius.unmap();
	holder->velocity.unbind();
	holder->mass.unbind();
	holder->count.unbind();
}

Phx::Phx(MEM::MISC::PlanetHolder *h)
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
