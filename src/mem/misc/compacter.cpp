#include "compacter.h"
#include "holder_kernels.h"

using namespace MEM::MISC;

const unsigned MAX_32BIT_MULTIPLIER = 3;

Compacter::Compacter( BufferCu<unsigned> *_mask )
	: size( _mask->getLen() )
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
	ASSERT( elem_size / sizeof(unsigned) <= MAX_32BIT_MULTIPLIER );
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
	
	CUDPPResult result = cudppPlan( &scanHandle, cfg, MAX_32BIT_MULTIPLIER * size, 1, 0 );
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

size_t Compacter::compact( IdxChangeSet *idx_change_set )
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
			updateIndices( &indices, idx_change_set );
		}
		else
		{
			stretch( mask, &wide_mask, it->first );
			compactLoop( &wide_mask, &indices, it->second );
		}
	}
	return newSize;
}

void Compacter::updateIndices( BufferCu<unsigned> *indices, IdxChangeSet *idx_change_set )
{
	for( IdxChangeSet::iterator it = idx_change_set->begin();
		it != idx_change_set->end(); ++it )
	{
		unsigned old_idx = it->first;
		if( mask->getAt( old_idx ) )
		{
			it->second = indices->getAt( old_idx );
		}
		else
		{
			it->second = -1;
		}
	}
}

