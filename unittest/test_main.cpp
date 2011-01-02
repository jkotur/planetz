#include <cppunit/ui/text/TestRunner.h>

#include <GL/glew.h>
#include <SDL/SDL.h>

#include "phx/phx_test.h"
#include "gfx/shader_test.h"
#include "util/config_test.h"
#include "mem/misc/buffer_test.h"
#include "mem/misc/holder_test.h"

void collectTests( CppUnit::TextUi::TestRunner &runner )
{
	runner.addTest( PhxTest::suite() );
	runner.addTest( CfgTest::suite() );
	runner.addTest( BufTest::suite() );
	runner.addTest( ShaderTest::suite() );
	runner.addTest( HolderTest::suite() );
}

int main(int argc, char **argv)
{
	if( SDL_Init(SDL_INIT_VIDEO|SDL_INIT_TIMER) ) {
		fprintf(stderr,"SDL error occurred: %s\n",SDL_GetError());
		return 1;
	}
	Uint32 flags = SDL_OPENGL|SDL_HWSURFACE|SDL_DOUBLEBUF;
	if( !SDL_SetVideoMode(600,400, 0, flags) ) {
		fprintf(stderr,"Cannot set video mode: %s\n",SDL_GetError() );
		return 2;
	}

        GLenum err = glewInit();
	if( GLEW_OK != err )
	{
		fprintf(stderr,"Glew error: %s\n", glewGetErrorString(err));
		return 3;
	}

	if( !glewIsSupported("GL_VERSION_3_2") )
	{
		fprintf(stderr,"OpenGL 3.2 is not supported.\n");
		return 4;
	}

	CppUnit::TextUi::TestRunner runner;
	collectTests( runner );
	runner.run();

	SDL_Quit();

	exit(0);
}

