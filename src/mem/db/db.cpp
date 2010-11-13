#include "db.h"

using namespace MEM;

Database::Database( const std::string &cs )
	: connection_string( cs )
{
}

Database::~Database()
{
}

bool Database::save( const ITable& t )
{
	std::string init = t.getCreationString();
	return sendSaveString( init + t.getSaveString() );
}

bool Database::load( ITable& t )
{
	return sendLoadString( t.getLoadString(), t );
}
