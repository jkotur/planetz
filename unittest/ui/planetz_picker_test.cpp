#include "planetz_picker_test.h"

#include <cppunit/extensions/HelperMacros.h>

#include "ui/planetz_picker.h"

#include "mem/misc/gfx_planet_factory.h"

#include "cuda/math.h"

PzPTest::PzPTest()
	: holder(0) , gpf(&holder) , picker(&gpf,3,3,640,480)
{
}

void PzPTest::setUp()
{
	holder.resize(3);

	float3 * pos = holder.pos.map( MEM::MISC::BUF_H );
	pos[0] = make_float3( 0 , 0 , 1 );
	pos[1] = make_float3( 0 , 0 ,-1 );
	pos[2] = make_float3( 0 , 0 ,-2 );
	holder.pos.unmap();

	float * rad = holder.radius.map( MEM::MISC::BUF_H );
	rad[0] = 1;
	rad[1] = 1;
	rad[2] = 1;
	holder.radius.unmap();
}

void PzPTest::tearDown()
{
}

void PzPTest::pickTest()
{
	picker.render(320,240);

	CPPUNIT_ASSERT_EQUAL( 1 , picker.getId() );

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt( 0,0,0 , 0,0,1 , 0,1,0 );

	picker.render(320,240);

	CPPUNIT_ASSERT_EQUAL( 0 , picker.getId() );

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt( 0,0,-5 , 0,0,0 , 0,-1,0 );

	picker.render(320,240);

	CPPUNIT_ASSERT_EQUAL( 2 , picker.getId() );

	picker.render(0,0);

	CPPUNIT_ASSERT_EQUAL(-1 , picker.getId() );

}

