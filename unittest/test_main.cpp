#include "phx/phx_test.h"
#include <cppunit/ui/text/TestRunner.h>

void collectTests( CppUnit::TextUi::TestRunner &runner )
{
	runner.addTest( PhxTest::suite() );
}

int main(int argc, char **argv)
{
	CppUnit::TextUi::TestRunner runner;
	collectTests( runner );
	runner.run();
	return 0;
}
