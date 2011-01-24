#ifndef _DB_MATERIALS_ROW_H_
#define _DB_MATERIALS_ROW_H_

#include "row.h"

namespace MEM
{
/**
 * @brief Klasa definiująca wiersz z materiałami.
 */
class MaterialsRow : public Row
{
	public:
		MaterialsRow();
		virtual ~MaterialsRow();

		virtual uint8_t size() const;
		virtual void setCell( unsigned idx, const std::string &val );

		typedef struct { float r, g, b; } color;
		unsigned id;
		color col;
		float ke;
		float ka;
		float kd;
		float alpha;
		color atm;
		float atmDensity;
		float atmRadius;
		unsigned texId;

	protected:
		virtual std::string getTableName() const;
		virtual std::string getCellNames() const;
		virtual std::string getCellDefs() const;
		virtual std::string getCellValues() const;
};
}

#endif // _DB_MATERIALS_ROW_H_
