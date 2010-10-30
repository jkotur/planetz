#include "phx.h"

using namespace GPU;

class Phx::CImpl
{
	public:
		CImpl(Holder *h);
		virtual ~CImpl();

		void compute();

	private:
		Holder *holder;
}

Phx::CImpl::CImpl(Holder *h)
	: holder(h)
{
}

Phx::CImpl::~CImpl()
{
}

void Phx::CImpl::compute(unsigned n)
{
	//map_buffers();
	//run_clusters();
	//for(unsigned i = 0; i < n; ++i)
	//{
	//	run_nbodies();
	//}
	//unmap_buffers();
}

Phx::Phx(Holder *h)
	: impl( new CImpl(h) )
{
}

Phx::~Phx()
{
	delete CImpl;
}

void Phx::compute(unsigned n)
{
	impl->compute(n);
}
