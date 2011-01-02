#ifndef __SHADER_TEST_H__

#define __SHADER_TEST_H__



#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>

class ShaderTest: public CppUnit::TestFixture
{
	public:
		void setUp();
		void tearDown();
		void testKMeans();

		void compileShadersTest();
		void linkProgramTest();

	private:
		CPPUNIT_TEST_SUITE( ShaderTest );
			CPPUNIT_TEST( compileShadersTest );
			CPPUNIT_TEST( linkProgramTest );
		CPPUNIT_TEST_SUITE_END();
};

#endif /* __SHADER_TEST_H__ */

