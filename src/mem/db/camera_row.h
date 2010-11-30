#ifndef _DB_CAMERA_ROW_H_
#define _DB_CAMERA_ROW_H_

#include "row.h"

namespace MEM
{
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

			float xcoord;
			float ycoord;
			float zcoord;

			float xlook;
			float ylook;
			float zlook;

			float xup;
			float yup;
			float zup;

		private:
			static const std::string save_string;
			static const std::string load_string;
			static const std::string creation_string;
	};
}

#endif // _DB_CAMERA_ROW_H_

