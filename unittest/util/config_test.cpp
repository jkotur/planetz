#include "config_test.h"

#include <cppunit/extensions/HelperMacros.h>

#include <ctime>
#include <cstdlib>

using namespace std;

CfgTest::CfgTest()
{
	srand(time(NULL));
}

void CfgTest::setUp()
{
}

void CfgTest::tearDown()
{
	cfg.clear();
}

void CfgTest::simpleSetAndGet()
{
	int i = rand();

	cfg.set( "int val" , i );

	CPPUNIT_ASSERT_EQUAL( i , cfg.get<int>( "int val" ) );
}

void CfgTest::failGet()
{
	int i = rand();

	cfg.set( "int val" , i );

	CPPUNIT_ASSERT_EQUAL( std::string() , cfg.get<std::string>( "int val" ) );
	CPPUNIT_ASSERT_EQUAL( 0.0           , cfg.get<double>     ( "int val" ) );
	CPPUNIT_ASSERT_EQUAL( 0.0f          , cfg.get<float>      ( "int val" ) );
}

void CfgTest::emptyGet()
{
	CPPUNIT_ASSERT_EQUAL( 0 , cfg.get<int>("random text") );
	CPPUNIT_ASSERT_EQUAL( std::string() , cfg.get<std::string>( "nope" ) );
}

