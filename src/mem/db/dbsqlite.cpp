#include <sqlite3.h>
#include <debug/routines.h>
#include "dbsqlite.h"
#include "row.h"
#include "util/logger.h"

using namespace MEM;

class DBSqlite::CImpl
{
	public:
		CImpl(DBSqlite *);
		~CImpl();

		bool sendSaveString( const std::string& sql );
		bool sendLoadString( const std::string& sql, ITable &t );
	private:
		void fill_table( ITable &t, char **source, int rows, int cols );

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
	DBGPUT(int e = )sqlite3_open( owner->connection_string.c_str() , &dbsocket );
	ASSERT( SQLITE_OK == e );
}

DBSqlite::CImpl::~CImpl()
{
	DBGPUT(int e = )sqlite3_close( dbsocket );
	ASSERT( SQLITE_OK == e );
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
		fill_table( t, table, rows, cols );
	}
	sqlite3_free_table( table );
	return SQLITE_OK == e;
}

void DBSqlite::CImpl::fill_table( ITable &t, char **source, int rows, int cols )
{
	for( int i = 1; i <= rows; ++i )
	{
		Row *r = t.insert_new();
		ASSERT( r->size() == cols );
		for( int j = 0; j < cols; ++j )
			r->setCell( j, std::string( source[i * cols + j] ) );
	}
}
