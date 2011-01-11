#include "holder.h"
#include "holder_kernels.h"
#include "cudpp.h"
#include "util/timer/timer.h"
#include <map>
#include <list>

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

class Compacter
{
	public:
		Compacter( unsigned _size, BufferCu<unsigned> *_mask );
		virtual ~Compacter();
		
		void add( void *d_data, unsigned elem_size );

		size_t compact();

	private:
		typedef std::list<void*> PtrList;
		typedef std::map<unsigned, PtrList> PtrListMap;

		void createScanHandle();
		unsigned compactLoop( BufferCu<unsigned> *mask, BufferCu<unsigned> *indices, const PtrList& list );
		void scan( BufferCu<unsigned> *in, BufferCu<unsigned> *out );

		CUDPPHandle scanHandle;
		unsigned size;
		BufferCu<unsigned> *mask;

		PtrListMap map;
};

Compacter::Compacter( unsigned _size, BufferCu<unsigned> *_mask )
	: size( _size )
	, mask( _mask )
{
	createScanHandle();
}

Compacter::~Compacter()
{
	cudppDestroyPlan( scanHandle );
}

void Compacter::add( void *d_data, unsigned elem_size )
{
	ASSERT( elem_size % sizeof(unsigned) == 0 );
	map[ elem_size / sizeof(unsigned) ].push_back( d_data );
}

void Compacter::scan( BufferCu<unsigned> *in, BufferCu<unsigned> *out )
{
	cudppScan( scanHandle, out->d_data(), in->d_data(), in->getLen() );
}

void Compacter::createScanHandle()
{
	CUDPPConfiguration cfg;
	cfg.datatype = CUDPP_UINT;
	cfg.algorithm = CUDPP_SCAN;
	cfg.options = CUDPP_OPTION_FORWARD  | CUDPP_OPTION_EXCLUSIVE;
	cfg.op = CUDPP_ADD;
	
	CUDPPResult result = cudppPlan( &scanHandle, cfg, 3 * size, 1, 0 );
	ASSERT( result == CUDPP_SUCCESS );
}

unsigned Compacter::compactLoop( BufferCu<unsigned> *mask, BufferCu<unsigned> *indices, const PtrList& list )
{
	scan( mask, indices );
	unsigned newSize = 0;
	for( PtrList::const_iterator it = list.begin(); it != list.end(); ++it )
	{
		newSize = reassign( *it, indices, mask );
	}
	return newSize;
}

size_t Compacter::compact()
{
	BufferCu<unsigned> indices;
	BufferCu<unsigned> wide_mask;
	unsigned newSize = 0;
	for( PtrListMap::iterator it = map.begin(); it != map.end(); ++it )
	{
		indices.resize( size * it->first );
		if( 1 == it->first ) // no stretching needed
		{
			newSize = compactLoop( mask, &indices, it->second );	
		}
		else
		{
			stretch( mask, &wide_mask, it->first );
			compactLoop( &wide_mask, &indices, it->second );
		}
	}
	return newSize;
}

namespace MEM
{
namespace MISC
{
void __filter( PlanetHolder *what, BufferCu<unsigned> *how )
{
	Compacter c( what->size(), how );
	c.add( what->model.map(BUF_CU), sizeof(int) );
	c.add( what->texId.map(BUF_CU), sizeof(int) );
	c.add( what->light.map(BUF_CU), sizeof(float) );
	c.add( what->radius.map(BUF_CU), sizeof(float) );
	c.add( what->mass.d_data(), sizeof(float) );
	c.add( what->atm_data.map(BUF_CU), sizeof(float2) );
	c.add( what->atm_color.map(BUF_CU), sizeof(float3) );
	c.add( what->pos.map(BUF_CU), sizeof(float3) );
	c.add( what->velocity.d_data(), sizeof(float3) );
	
	unsigned newSize = c.compact();
	cudaMemcpy( what->count.map(BUF_CU), &newSize, sizeof(unsigned), cudaMemcpyHostToDevice );

	what->count.unmap();
	what->model.unmap();
	what->texId.unmap();
	what->light.unmap();
	what->radius.unmap();
	what->atm_data.unmap();
	what->atm_color.unmap();
	what->pos.unmap();
}
}
}

