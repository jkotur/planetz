#ifndef __PLANETZ_PICKER_TEST_H__

#define __PLANETZ_PICKER_TEST_H__

#include "gfx/planetz_picker.h"
#include "mem/misc/holder.h"

#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>

class PzPTest : public CppUnit::TestFixture
{
	public:
		PzPTest();

		void setUp();
		void tearDown();
		void testKMeans();

		void pickTest();

	private:
		CPPUNIT_TEST_SUITE( PzPTest );
			CPPUNIT_TEST( pickTest );
		CPPUNIT_TEST_SUITE_END();

		MEM::MISC::PlanetHolder holder;
		MEM::MISC::GfxPlanetFactory gpf;
		GFX::PlanetzPicker picker;
};

#endif /* __PLANETZ_PICKER_TEST_H__ */

