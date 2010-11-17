#ifndef __SAVER_H__

#define __SAVER_H__

#include <string>
#include "misc/holder.h"

namespace MEM
{
	class Saver
	{
	public:
		Saver();
		virtual ~Saver();

		void save( const MISC::CpuPlanetHolder *source, const std::string& path );
		MISC::CpuPlanetHolder *load( const std::string& path );
	};
}

#endif /* __SAVER_H__ */

