#include "options.h"

#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#include "util/logger.h"

using std::string;
using std::vector;

namespace po = boost::program_options;

Options::Options()
	: converted(false)
	, desc("Opcje")
{
	desc.add_options()
		("help,h", "produce help message")
		("resolution,r", po::value< vector<int> >()->multitoken(),
		 "window resolution")
		("fullscreen,f", po::value<bool>(),
		 "tryb pełnoekranowy")
		("deffered.textures", po::value<bool>(),
		 "teksturowanie")
		("deffered.lighting", po::value<bool>(),
		 "oświetlenie")
		("deffered.lights_range", po::value<bool>(),
		 "zasięg świateł")
		("deffered.normals", po::value<bool>(),
		 "normalne")
		("deffered.brightness", po::value<float>(),
		 "jasność światła otoczenia")
		("trace.enable", po::value<bool>(),
		 "włączenie lub wyłączenie śladu")
		("trace.visible", po::value<bool>(),
		 "kontroluje wyświetlanie śladu")
		("trace.frequency", po::value<double>(),
		 "częstotliwość stawiania śladu (w sekundach)")
		("trace.length", po::value<unsigned>(),
		 "ilość śladów które zostawia planeta")
		;
}

Options::~Options()
{
}

void Options::addCmdLine( int argc , const char**argv )
{
	converted = false;

	try {
		po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
	} catch( po::error& e ) {
		log_printf(_ERROR,"Cannot parse command line parameter: %s. Fallback to default.\n", e.what() );
	}
}

void Options::addCfgFile( const char*file )
{
	converted = false;

	std::ifstream fs(file);

	try {
		po::store(po::parse_config_file(fs, desc), vm);
	} catch( po::error& e ) {
		log_printf(_ERROR,"Cannot parse config file parameter: %s. Fallback to default.\n", e.what() );
	}
}

template<typename T>
void vmtocfg( Config& to , po::variables_map& from , const char*str )
{
	if( from.count(str) ) to.set<T>( str , from[str].as<T>() );
}

void vmtocfg_present( Config& to , po::variables_map& from , const char*str )
{
	to.set<bool>( str , from.count(str) );
}

const Config& Options::getCfg()
{
	if( converted ) return cfg;

	po::notify(vm);

	vmtocfg_present      ( cfg , vm , "help"       );
	vmtocfg<vector<int> >( cfg , vm , "resolution" );
	vmtocfg<bool>        ( cfg , vm , "fullscreen" );

	vmtocfg<bool>        ( cfg , vm , "deffered.textures"       );
	vmtocfg<bool>        ( cfg , vm , "deffered.lights_range"   );
	vmtocfg<bool>        ( cfg , vm , "deffered.lighting"   );
	vmtocfg<bool>        ( cfg , vm , "deffered.normals"   );
	vmtocfg<float>       ( cfg , vm , "deffered.brightness"   );

	vmtocfg<bool>        ( cfg , vm , "trace.enable"    );
	vmtocfg<bool>        ( cfg , vm , "trace.visible"   );
	vmtocfg<double>      ( cfg , vm , "trace.frequency" );
	vmtocfg<unsigned>    ( cfg , vm , "trace.length"    );

	converted = true;
	return cfg;
}

std::string Options::getHelp()
{
	std::stringstream ss;
	ss << desc << std::endl;
	return ss.str();
}

