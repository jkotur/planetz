#ifndef _DB_SQLITE_H_
#define _DB_SQLITE_H_

#include "db.h"

namespace MEM
{
	/**
	 * @brief Implementacja Database dla bazy danych sqlite3.
	 */
	class DBSqlite : public Database
	{
		public:
			/**
			 * @brief Tworzy połączenie z bazą danych.
			 *
			 * @param cs Nazwa pliku z bazą.
			 */
			DBSqlite( const std::string& cs );
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
