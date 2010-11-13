#ifndef _PLANET_ROW_H_
#define _PLANET_ROW_H_

#include "row.h"

namespace MEM
{
	class PlanetRow : public Row
	{
		public:
			PlanetRow();
			virtual ~PlanetRow();

			virtual std::string getSaveString() const;
			virtual std::string getLoadString() const;
			virtual std::string getCreationString() const;

			virtual uint8_t size() const;

		public: // You are allowed to modify these freely. Remember, however, to save them in db later.
			float xcoord;
			float ycoord;
			float zcoord;
			float radius;
			float mass;
			float xvel;
			float yvel;
			float zvel;
			uint8_t model_id;

		private:
			static const std::string save_string;
			static const std::string load_string;
			static const std::string creation_string;
	};
}

#endif // _PLANET_ROW_H_
