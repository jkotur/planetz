#ifndef _DB_TEXTURES_ROW_H_
#define _DB_TEXTURES_ROW_H_

#include "row.h"
#include <string>

namespace MEM
{
/**
 * @brief Klasa definiująca wiersz z materiałami.
 */
class TexturesRow : public Row
{
	public:
		TexturesRow();
		virtual ~TexturesRow();

		virtual uint8_t size() const;
		virtual void setCell( unsigned idx, const std::string &val );

		unsigned id;
		std::string path;

	protected:
		virtual std::string getTableName() const;
		virtual std::string getCellNames() const;
		virtual std::string getCellDefs() const;
		virtual std::string getCellValues() const;
};
}

#endif // _DB_TEXTURES_ROW_H_

