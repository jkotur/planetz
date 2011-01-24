#ifndef _DB_CAMERA_ROW_H_
#define _DB_CAMERA_ROW_H_

#include "row.h"

namespace MEM
{
	/**
	 * @brief Klasa definiująca wiersz kamery z bazy danych.
	 */
	class CameraRow : public Row
	{
		public:
			CameraRow();
			virtual ~CameraRow();

			virtual uint8_t size() const;
			virtual void setCell( unsigned idx, const std::string &val );

			/**
			 * @brief Macierz przekształceń kamery.
			 */
			float matrix[16];

		protected:
			virtual std::string getTableName() const;
			virtual std::string getCellNames() const;
			virtual std::string getCellDefs() const;
			virtual std::string getCellValues() const;
	};
}

#endif // _DB_CAMERA_ROW_H_

