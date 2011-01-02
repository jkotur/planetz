#include "buffer_test.h"

#include <cppunit/extensions/HelperMacros.h>

BufTest::BufTest()
{
	for( int i=0 ; i<20 ; i++ )
		data[i] = i;
}

void BufTest::setUp()
{
	buf.resize( 10 , data );
}

void BufTest::tearDown()
{
}

void BufTest::resizeTest()
{
	buf.resize( 5 , data );

	int* data2 = buf.map( BUF_H );
	for( unsigned i=0 ; i<buf.getLen() ; i++ )
		CPPUNIT_ASSERT_EQUAL( data[i] , (int)i );
	buf.unmap();

	buf.resize( 20 , data );

	data2 = buf.map( BUF_H );
	for( unsigned i=0 ; i<buf.getLen() ; i++ )
		CPPUNIT_ASSERT_EQUAL( data[i] , (int)i );
	buf.unmap();

	buf.resize( 5 , data );

	data2 = buf.map( BUF_H );
	for( unsigned i=0 ; i<buf.getLen() ; i++ )
		CPPUNIT_ASSERT_EQUAL( data[i] , (int)i );
	buf.unmap();

}

void BufTest::dataTest()
{
	int* data2 = buf.map( BUF_H );
	for( int i=0 ; i<10 ; i++ )
		CPPUNIT_ASSERT_EQUAL( data[i] , data2[i] );
	buf.unmap();
}

void BufTest::hreadwriteTest()
{
	int* data2 = buf.map( BUF_H );
	for( int i=0 ; i<10 ; i++ )
		data2[i] = i*i;
	buf.unmap();

	int* data3 = buf.map( BUF_H );
	for( int i=0 ; i<10 ; i++ )
		CPPUNIT_ASSERT_EQUAL( data3[i] , i*i );
	buf.unmap();
}

void BufTest::cudaTest()
{
	buf.resize( 5 );
	buf.map( BUF_H );
	buf.unmap();
	buf.map( BUF_CU );
	buf.unmap();
	buf.resize( 20 );
	buf.map( BUF_H );
	buf.unmap();
	buf.map( BUF_CU );
	buf.unmap();
}

