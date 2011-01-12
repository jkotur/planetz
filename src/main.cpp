#ifdef _WIN32
# include <windows.h>
#endif

#include <vector>

#include "constants.h"

#include "window.h"
#include "options.h"
#include "application.h"

int main (int argc, char const* argv[])
{
	log_add(LOG_STREAM(stderr),LOG_PRINTER(std::vfprintf));

//        log_set_lev( INFO );

	Options opt;

	opt.addCmdLine( argc , argv );
	opt.addCfgFile( DATA("planetz.cfg").c_str() );

	Config cfg = opt.getCfg();

	if( cfg.get<bool>("help") ) {
		log_printf(INFO,"%s",opt.getHelp().c_str());
		return 0;
	}

	Window win( cfg.get<std::vector<int> >("resolution") ,
		    cfg.get<bool>("fullscreen") );

	if( !win ) return win.getErr();

	Application app( win , cfg );

	if( !app.init() )
		return 1;

#ifndef _RELEASE
	app.test();
#endif

	app.main_loop();

	return 0;
}

