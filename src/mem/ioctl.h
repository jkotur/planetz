#ifndef _IOCTL_H_
#define _IOCTL_H_

#include <string>

#include "mem/misc/materials_manager.h"
#include "mem/misc/textures_manager.h"

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
	void loadMaterials( MISC::Materials* dest , const std::string & path );
	void loadTextures( MISC::Textures* dest , const std::string & path );

private:
	class Impl;
	Impl* impl;
};
}

#endif // _IOCTL_H_
