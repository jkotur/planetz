#ifndef _DB_CAMERA_ROW_H_
#define _DB_CAMERA_ROW_H_

#include "row.h"
#include "util/vector.h"

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

			virtual std::string getSaveString() const;
			virtual std::string getLoadString() const;
			virtual std::string getCreationString() const;

			virtual uint8_t size() const;
			virtual void setCell( unsigned idx, const std::string &val );

			/**
			 * @brief Położenie kamery.
			 */
			Vector3 coords;

			/**
			 * @brief Kierunek patrzenia kamery.
			 */
			Vector3 lookat;

			/**
			 * @brief Wektor góry kamery.
			 */
			Vector3 up;

		private:
			static const std::string save_string;
			static const std::string load_string;
			static const std::string creation_string;
	};
}

#endif // _DB_CAMERA_ROW_H_

