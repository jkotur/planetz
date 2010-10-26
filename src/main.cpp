#ifdef _WIN32
# include <windows.h>
#endif

#include "application.h"

int main (int argc, char const* argv[])
{

	Application app;

	if( !app.init() )
		return 1;

	app.main_loop();

	return 0;
}

