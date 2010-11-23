#ifndef _ROW_H_
#define _ROW_H_

#include <string>

namespace MEM
{
	typedef unsigned char uint8_t; // TODO: move it somewhere else
	class Row
	{
		public:
			Row();
			virtual ~Row();

			virtual std::string getSaveString() const = 0;
			virtual std::string getLoadString() const = 0;
			virtual std::string getCreationString() const = 0;

			/// @brief Number of cells in a row
			virtual uint8_t size() const = 0;
			virtual void setCell( unsigned idx, const std::string& val ) = 0;
	};
}

#endif // _ROW_H_
