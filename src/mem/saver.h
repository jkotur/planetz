#ifndef __SAVER_H__

#define __SAVER_H__

#include <string>

namespace MEM
{
namespace MISC
{
	class SaverParams;
}
	class Saver
	{
	public:
		Saver();
		virtual ~Saver();

		void save( const MISC::SaverParams *source, const std::string& path );
		void load( MISC::SaverParams *dest, const std::string& path );
	};
}

#endif /* __SAVER_H__ */

