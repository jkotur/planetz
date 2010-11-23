#ifndef _TABLE_INTERFACE_H_
#define _TABLE_INTERFACE_H_

#include <string>

namespace MEM
{
	class Row;

	class ITable
	{
		public:
			virtual std::string getSaveString() const = 0;
			virtual std::string getLoadString() const = 0;
			virtual std::string getCreationString() const = 0;

			virtual Row* insert_new() = 0;
	};
}

#endif // _TABLE_INTERFACE_H_

