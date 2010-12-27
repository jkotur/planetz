#ifndef _PHX_TEST_H_
#define _PHX_TEST_H_

#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>

#ifndef NVCC
struct float3;
#endif
namespace MEM
{
namespace MISC
{
	template<class T>
	class BufferGl;
	template<class T>
	class BufferCu;
}
}

namespace PHX
{
	class Clusterer;
}

class PhxTest: public CppUnit::TestFixture
{
	public:
		void setUp();
		void tearDown();
		void testKMeans();

	private:
		CPPUNIT_TEST_SUITE( PhxTest );
			CPPUNIT_TEST( testKMeans );
		CPPUNIT_TEST_SUITE_END();

		MEM::MISC::BufferGl<float3> *pos;
		MEM::MISC::BufferCu<unsigned> *shuffle;
		MEM::MISC::BufferCu<unsigned> *counts;
		MEM::MISC::BufferCu<float3> *centers;
		PHX::Clusterer* c;
};

#endif // _PHX_TEST_H_
