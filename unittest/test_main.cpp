#include <cppunit/ui/text/TestRunner.h>

#include "phx/phx_test.h"
#include "util/config_test.h"

void collectTests( CppUnit::TextUi::TestRunner &runner )
{
	runner.addTest( PhxTest::suite() );
	runner.addTest( CfgTest::suite() );
}

int main(int argc, char **argv)
{
	CppUnit::TextUi::TestRunner runner;
	collectTests( runner );
	runner.run();
	return 0;
}

