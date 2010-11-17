#include "ioctl.h"
#include "saver.h"
#include "constants.h"

using namespace MEM;

class IOCtl::Impl
{
	public:
		void save( const MISC::CpuPlanetHolder *source, const std::string& path );
		MISC::CpuPlanetHolder *load( const std::string& path );

	private:
		Saver s;
		
};

IOCtl::IOCtl()
{
	impl = new IOCtl::Impl();
}

IOCtl::~IOCtl()
{
	delete impl;
}

void IOCtl::save( const MISC::CpuPlanetHolder *source, const std::string &path )
{
	impl->save( source, path );
}

MISC::CpuPlanetHolder* IOCtl::load( const std::string &path )
{
	return impl->load( path );
}

void IOCtl::Impl::save( const MISC::CpuPlanetHolder *source, const std::string &path )
{
	s.save( source, path );
}

MISC::CpuPlanetHolder* IOCtl::Impl::load( const std::string &path )
{
	return s.load( path );
}
