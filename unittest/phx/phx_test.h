#ifndef _PHX_TEST_H_
#define _PHX_TEST_H_

#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>


class PhxTest: public CppUnit::TestFixture
{
	public:
		void setUp();
		void tearDown();
		void testWhatever();

	private:
		
};

#endif // _PHX_TEST_H_
