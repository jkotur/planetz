#include "phx/phx_test.h"
#include <cppunit/TestCaller.h>
#include <cppunit/TestResult.h>

int main(int argc, char **argv)
{
	CppUnit::TestCaller<PhxTest> test( "testWhatever", &PhxTest::testWhatever );
	CppUnit::TestResult result;
	test.run( &result );

	return 0;
}
