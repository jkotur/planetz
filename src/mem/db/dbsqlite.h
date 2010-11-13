#ifndef _DB_SQLITE_H_
#define _DB_SQLITE_H_

#include "db.h"

namespace MEM
{
	class DBSqlite : public Database
	{
		public:
			DBSqlite( const std::string& );
			virtual ~DBSqlite();

		protected:
			virtual bool sendSaveString( const std::string& sql );
			virtual bool sendLoadString( const std::string& sql, ITable& t );

		private:
			class CImpl;
			friend class CImpl;
			CImpl *impl;
	};
}

#endif // _DB_SQLITE_H_
