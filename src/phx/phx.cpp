#include "phx.h"

using namespace CPU;

class Phx::CImpl
{
	public:
		CImpl(GPU::Holder *h);
		virtual ~CImpl();

		void compute(unsigned n);

	private:
		void map_buffers();
		void unmap_buffers();

		GPU::Holder *holder;
};

Phx::CImpl::CImpl(GPU::Holder *h)
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
	holder->planet_pos.map( GPU::BUF_CU );
	holder->planet_radius.map( GPU::BUF_CU );
	holder->planet_velocity.bind();
	holder->planet_mass.bind();
	holder->planet_count.map( GPU::BUF_CU );
}

void Phx::CImpl::unmap_buffers()
{
	holder->planet_pos.unmap();
	holder->planet_radius.unmap();
	holder->planet_velocity.unbind();
	holder->planet_mass.unbind();
	holder->planet_count.unbind();
}

Phx::Phx(GPU::Holder *h)
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
