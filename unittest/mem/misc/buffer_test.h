#ifndef __BUFFER_TEST_H__

#define __BUFFER_TEST_H__


#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>

#include "mem/misc/buffer_gl.h"

using namespace MEM::MISC;

class BufTest : public CppUnit::TestFixture
{
	public:
		BufTest();

		void setUp();
		void tearDown();

		void resizeTest();
		void resizePreserveTest();
		void dataTest();
		void hreadwriteTest();
		void cudaTest();

	private:
		int data[20];
		BufferGl<int> buf;
		BufferCu<int> bufCu;

		CPPUNIT_TEST_SUITE( BufTest );
			CPPUNIT_TEST( resizeTest );
			CPPUNIT_TEST( resizePreserveTest );
			CPPUNIT_TEST( dataTest   );
			CPPUNIT_TEST( hreadwriteTest );
			CPPUNIT_TEST( cudaTest   );
		CPPUNIT_TEST_SUITE_END();
};

#endif /* __BUFFER_TEST_H__ */

