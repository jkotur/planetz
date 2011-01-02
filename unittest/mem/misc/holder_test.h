#ifndef __HOLDER_TEST_H__
#define __HOLDER_TEST_H__

#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>

#include "mem/misc/holder.h"

class HolderTest : public CppUnit::TestFixture
{
	public:
		HolderTest();

		void setUp();
		void tearDown();

		void filterTest();

	private:
		void initData( unsigned i );

		MEM::MISC::PlanetHolder holder;

		CPPUNIT_TEST_SUITE( HolderTest );
			CPPUNIT_TEST( filterTest );
		CPPUNIT_TEST_SUITE_END();
};

#endif // __HOLDER_TEST_H__
