#include "holder_test.h"

HolderTest::HolderTest()
{
}

void HolderTest::setUp()
{
	initData( 50 );
}

void HolderTest::tearDown()
{
}

float3 make_float3( float x, float y, float z )
{
	float3 v;
	v.x = x;
	v.y = y;
	v.z = z;
	return v;
}

void HolderTest::filterTest()
{
	MEM::MISC::BufferCu<unsigned> mask( 50 );
	mask.bind();
	for( unsigned i = 0; i < 50; ++i )
		mask.h_data()[i] = i % 2;
	mask.unbind();

	holder.filter( &mask );

	CPPUNIT_ASSERT_EQUAL( holder.size(), (size_t)25 );

	float3 *h_pos = holder.pos.map(MEM::MISC::BUF_H);
	holder.velocity.bind();
	float3 *h_vel = holder.velocity.h_data();
	for( unsigned i = 0; i < 25; ++i )
	{
		float3 v = make_float3( 2 * i + 1, 4 * i + 2, 6 * i + 3 );
		CPPUNIT_ASSERT_EQUAL( v.x, h_pos[i].x );
		CPPUNIT_ASSERT_EQUAL( v.y, h_pos[i].y );
		CPPUNIT_ASSERT_EQUAL( v.z, h_pos[i].z );
		CPPUNIT_ASSERT_EQUAL( v.x, h_vel[i].x );
		CPPUNIT_ASSERT_EQUAL( v.y, h_vel[i].y );
		CPPUNIT_ASSERT_EQUAL( v.z, h_vel[i].z );
	}
	holder.velocity.unbind();
	holder.pos.unmap();
}

void setBuf( float3 *buf, unsigned size )
{
	for( unsigned i = 0; i < size; ++i )
		buf[i] = make_float3( i, 2 * i, 3 * i );
}

void HolderTest::initData( unsigned size )
{
	holder.resize( size );

	float3 *h_pos = holder.pos.map(MEM::MISC::BUF_H);
	setBuf( h_pos, size );
	holder.pos.unmap();

	holder.velocity.bind();
	float3 *h_vel = holder.velocity.h_data();
	setBuf( h_vel, size );
	holder.velocity.unbind();
}
