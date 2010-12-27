#include <cppunit/ui/text/TestRunner.h>
#include <GL/glew.h>
#include <GL/glut.h>

#include "phx/phx_test.h"
#include "util/config_test.h"

void collectTests( CppUnit::TextUi::TestRunner &runner )
{
	runner.addTest( PhxTest::suite() );
	runner.addTest( CfgTest::suite() );
}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutCreateWindow("test");

        GLenum err = glewInit();
	if( GLEW_OK != err )
	{
		printf("[GLEW] %s\n", glewGetErrorString(err));
		return false;
	}
	CppUnit::TextUi::TestRunner runner;
	collectTests( runner );
	runner.run();
	return 0;
}

