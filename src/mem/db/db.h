#ifndef _DB_H_
#define _DB_H_

#include <string>
#include <debug/routines.h>
#include "table_interface.h"

namespace MEM
{
	/// dumb db interface
	class Database
	{
		public:
			Database( const std::string & );
			virtual ~Database();

			bool save( const ITable& t );
			bool load( ITable& t );

		protected:
			virtual bool sendSaveString( const std::string& sql ) = 0;
			virtual bool sendLoadString( const std::string& sql, ITable& t ) = 0;

			const std::string connection_string;
	};
}

#endif // _DB_H 
