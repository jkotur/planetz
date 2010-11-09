#include "phx_test.h"
#include <iostream>

using namespace std;

void PhxTest::setUp()
{
	cout << "cze" << endl;
}

void PhxTest::tearDown()
{
	cout << "bye" << endl;
}

void PhxTest::testWhatever()
{
	cout << "testing..." << endl;
	CPPUNIT_ASSERT( true );
}
