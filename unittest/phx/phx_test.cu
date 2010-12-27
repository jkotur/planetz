#include "phx_test.h"
#include <iostream>
#include <cppunit/extensions/HelperMacros.h>
#include <kmeans.h>

using namespace std;
using namespace PHX;
using namespace MEM::MISC;

void PhxTest::setUp()
{
	/*pos = new BufferGl<float3>();
	c = new Clusterer( pos );*/
}

void PhxTest::tearDown()
{
/*
	pos->unmap();
	shuffle->unbind();
	counts->unbind();
	centers->unbind();
	delete c;
	delete pos;
	*/
}

namespace
{
float dist2( float3 l, float3 r )
{
	float dx = l.x - r.x,
		dy = l.y - r.y,
		dz = l.z - r.z;
	return dx * dx + dy * dy + dz * dz;
}
}

void PhxTest::testKMeans()
{
	// TODO: przetestować to przy pełnej implementacji k-means
	/*
	pos->resize( 8000 );
	float3* h_pos = pos->map( BUF_H );
	for( unsigned x = 0; x < 20; ++x )
		for( unsigned y = 0; y < 20; ++y )
			for( unsigned z = 0; z < 20; ++z )
			{
				h_pos[x + 20 * y + 20 * 20 * z]
					= make_float3( 100 * atan(x-10), 100 * atan(y-10), 100*atan(z-10) );
			}
	pos->map( BUF_GL );
	c->kmeans();
	
	unsigned k = c->getCount();
	shuffle = c->getShuffle();
	counts = c->getCounts();
	centers = c->getCenters();
	shuffle->bind();
	counts->bind();
	centers->bind();
	unsigned *h_shuffle = shuffle->h_data();
	unsigned *h_counts = counts->h_data();
	float3 *h_centers = centers->h_data();
	h_pos = pos->map( BUF_H );
	CPPUNIT_ASSERT( h_shuffle );
	CPPUNIT_ASSERT( h_counts );
	CPPUNIT_ASSERT( h_centers );
	CPPUNIT_ASSERT( h_pos );
	for( unsigned i = 0; i < pos->getLen(); ++i )
	{
		unsigned cluster = 0;
		while( h_shuffle[ i ] >= h_counts[ cluster ] )
			++cluster;
		CPPUNIT_ASSERT( cluster < k );
		for( unsigned j = 0; j < k; ++j )
		{
			if( j != cluster )
			{
				float dist2this = dist2( h_pos[ h_shuffle[ i ] ], h_centers[ cluster ] );
				float dist2other = dist2( h_pos[ h_shuffle[ i ] ], h_centers[ j ] );
				ASSERT( dist2other >= dist2this );
			}
		}
	}
	*/
}
