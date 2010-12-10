#ifndef _CFG_TEST_H_
#define _CFG_TEST_H_

#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>

#include "util/config.h"

class CfgTest : public CppUnit::TestFixture
{
	public:
		CfgTest();

		void setUp();
		void tearDown();

		void simpleSetAndGet();
		void failGet();
		void emptyGet();

	private:
		Config cfg;

		CPPUNIT_TEST_SUITE( CfgTest );
			CPPUNIT_TEST( simpleSetAndGet );
			CPPUNIT_TEST( failGet );
			CPPUNIT_TEST( emptyGet);
		CPPUNIT_TEST_SUITE_END();
};

#endif // _CFG_TEST_H_
