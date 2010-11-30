#ifndef _MEM_SAVER_PARAMS_H_
#define _MEM_SAVER_PARAMS_H_

#include "holder.h"

class Camera;

namespace MEM
{
namespace MISC
{
	class SaverParams
	{
		public:
			SaverParams(Camera *cam);
			virtual ~SaverParams(); // deletes planet_info

			CpuPlanetHolder *planet_info;
			Camera *cam_info;
	};
}
}

#endif // _MEM_SAVER_PARAMS_H_

