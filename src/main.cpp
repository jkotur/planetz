#ifdef _WIN32
# include <windows.h>
#endif

#include "constants.h"

#include "window.h"
#include "application.h"

int main (int argc, char const* argv[])
{
	Window w( BASE_W , BASE_H );

	if( !w ) return w.getErr();

	Application app( w );

	if( !app.init() )
		return 1;

#ifndef _RELEASE
	app.test();
#endif

	app.main_loop();

	return 0;
}

