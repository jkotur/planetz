#ifndef _IOCTL_H_
#define _IOCTL_H_

#include <string>
#include "misc/holder.h"

namespace MEM
{
	class IOCtl
	{
		public:
			IOCtl();
			virtual ~IOCtl();

			void save( const MISC::CpuPlanetHolder *source, const std::string& path );
			MISC::CpuPlanetHolder *load( const std::string& path );

		private:
			class Impl;
			Impl* impl;
	};
}

#endif // _IOCTL_H_
