#include "holder.h"
#include "holder_kernels.h"
#include "cudpp.h"

using namespace MEM::MISC;

ClusterHolder::ClusterHolder()
	: m_size(0)
{
}

ClusterHolder::~ClusterHolder()
{
}

void ClusterHolder::resize(size_t k_size, size_t n_size)
{
	centers.resize(k_size);
	masses.resize(k_size);
	assignments.resize(n_size);
	m_size = k_size;
}

size_t ClusterHolder::k_size() const
{
	return m_size;
}

CUDPPHandle __createFilterHandle( /*CUDPPDatatype type, */unsigned size )
// type jest chyba niepotrzebny przy tym rodzaju obliczeń? Zasadniczo i tak tylko przepisujemy dane z jednego miejsca do innego
{
	CUDPPConfiguration cfg;
	cfg.datatype = CUDPP_UINT; //type;
	cfg.algorithm = CUDPP_COMPACT;
	cfg.options = CUDPP_OPTION_FORWARD;
	cfg.op = CUDPP_ADD;
	
	CUDPPHandle handle;
	CUDPPResult result = cudppPlan(&handle, cfg, size, 1, 0 );
	ASSERT( result == CUDPP_SUCCESS );

	return handle;
}

void __compact( CUDPPHandle handle, void *buffer, void *tmp_buffer, unsigned *mask, unsigned size, BufferCu<size_t> *d_size )
{
	cudppCompact( handle, tmp_buffer, d_size->d_data(), buffer, mask, size );
	cudaMemcpy( buffer, tmp_buffer, d_size->getAt( 0 ) * 4, cudaMemcpyDeviceToDevice );
}
#define COMPACT(x) __compact( handle, x, tmp.d_data(), how->d_data(), how->getLen(), &new_size )

void __filter_4bytes( CUDPPHandle handle, PlanetHolder *what, BufferCu<unsigned> *how )
{
	// da się to w ogóle zrobić ładnie? :/
	ASSERT( sizeof(int) == 4 );
	ASSERT( sizeof(float) == 4 );

	BufferCu<int> tmp( how->getLen() );
	BufferCu<size_t> new_size( 1 );

	// int
	COMPACT( what->model.map(BUF_CU) ); what->model.unmap();
	COMPACT( what->texId.map(BUF_CU) ); what->texId.unmap();
	// float
	COMPACT( what->light.map(BUF_CU) ); what->light.unmap();
	COMPACT( what->radius.map(BUF_CU) ); what->radius.unmap();
	COMPACT( what->mass.d_data() );
	
	cudaMemcpy( what->count.map(BUF_CU), new_size.d_data(), sizeof(unsigned), cudaMemcpyDeviceToDevice );
	what->count.unmap();
}

void __filter_8bytes( CUDPPHandle handle, PlanetHolder *what, BufferCu<unsigned> *how )
{
	ASSERT( sizeof(float2) == 8 );

	BufferCu<int> tmp( how->getLen() );
	BufferCu<size_t> new_size( 1 );

	// float2
	COMPACT( what->atm_data.map(BUF_CU) ); what->atm_data.unmap();
}

void __filter_12bytes( CUDPPHandle handle, PlanetHolder *what, BufferCu<unsigned> *how )
{
	ASSERT( sizeof(float3) == 12 );

	BufferCu<int> tmp( how->getLen() );
	BufferCu<size_t> new_size( 1 );

	// float3
	COMPACT( what->atm_color.map(BUF_CU) ); what->atm_color.unmap();
	COMPACT( what->pos.map(BUF_CU) ); what->pos.unmap();
	COMPACT( what->velocity.d_data() );
}

namespace MEM
{
namespace MISC
{
void __filter( PlanetHolder *what, BufferCu<unsigned> *how )
{
	BufferCu<unsigned> wide;
	CUDPPHandle plan = __createFilterHandle( 3 * how->getLen() );
	
	__filter_4bytes( plan, what, how );
	stretch( how, &wide, 2 );
	__filter_8bytes( plan, what, &wide );
	stretch( how, &wide, 3 );
	__filter_12bytes( plan, what, &wide );

	cudppDestroyPlan( plan );
}
}
}
