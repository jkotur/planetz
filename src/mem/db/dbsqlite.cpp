#include "dbsqlite.h"
#include <sqlite3.h>
#include <debug/routines.h>

using namespace MEM;

class DBSqlite::CImpl
{
	public:
		CImpl(DBSqlite *);
		~CImpl();

		bool sendSaveString( const std::string& sql );
		bool sendLoadString( const std::string& sql, ITable &t );
	private:
		sqlite3 *dbsocket;
		DBSqlite *owner;
};

DBSqlite::DBSqlite( const std::string& cs )
	: Database( cs )
{
	impl = new CImpl(this);
}

DBSqlite::~DBSqlite()
{
	delete impl;
}

bool DBSqlite::sendSaveString( const std::string& sql )
{
	return impl->sendSaveString( sql );
}

bool DBSqlite::sendLoadString( const std::string& sql, ITable &t )
{
	return impl->sendLoadString( sql, t );
}

DBSqlite::CImpl::CImpl(DBSqlite *_owner)
	: owner( _owner )
{
	TODO("error handling");
	sqlite3_open( owner->connection_string.c_str() , &dbsocket );
}

DBSqlite::CImpl::~CImpl()
{
	TODO("error handling");
	sqlite3_close( dbsocket );
}

namespace 
{
	void log_n_free( char *err )
	{
		log_printf( _ERROR, "Database error: %s\n", err );
		sqlite3_free( err );
	}
}

bool DBSqlite::CImpl::sendSaveString( const std::string& sql )
{
	log_printf(DBG, "sending save string\n");
	char *err;
	int e = sqlite3_exec( dbsocket, sql.c_str(), NULL, NULL, &err );
	if( e != SQLITE_OK )
	{
		log_n_free( err );
	}

	log_printf(DBG, "sending save string done\n");
	return SQLITE_OK == e;
}

bool DBSqlite::CImpl::sendLoadString( const std::string& sql, ITable& t )
{
	char *err;
	char **table;
	int rows, cols;
	int e = sqlite3_get_table( dbsocket, sql.c_str(), &table, &rows, &cols, &err );
	if( e != SQLITE_OK )
	{
		log_n_free( err );
	}
	else
	{
		TODO("fill 't' with smth");
	}
	return SQLITE_OK == e;
}
