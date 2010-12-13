#ifndef __OPTIONS_H__

#define __OPTIONS_H__

#include <boost/program_options.hpp>
#include <string>

#include "util/config.h"

class Options {
public:
	Options();
	virtual ~Options();

	void addCmdLine( int argc , const char**argv );
	void addCfgFile( const char*file );

	const Config& getCfg();

	std::string getHelp();
	
private:
	bool converted;
	Config cfg;

	boost::program_options::variables_map vm;
	boost::program_options::options_description desc;
};


#endif /* __OPTIONS_H__ */

