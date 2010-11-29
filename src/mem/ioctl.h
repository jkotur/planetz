#ifndef _IOCTL_H_
#define _IOCTL_H_

#include <string>

namespace MEM
{
namespace MISC
{
	class SaverParams;
}
	class IOCtl
	{
		public:
			IOCtl();
			virtual ~IOCtl();

			void save( const MISC::SaverParams *source, const std::string& path );
			void load( MISC::SaverParams *dest, const std::string& path );

		private:
			class Impl;
			Impl* impl;
	};
}

#endif // _IOCTL_H_
