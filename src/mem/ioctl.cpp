#include "ioctl.h"
#include "saver.h"
#include "constants.h"

using namespace MEM;

class IOCtl::Impl
{
	public:
		void save( const MISC::SaverParams *source, const std::string& path );
		void load( MISC::SaverParams *dest, const std::string& path );

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

void IOCtl::save( const MISC::SaverParams *source, const std::string &path )
{
	impl->save( source, path );
}

void IOCtl::load( MISC::SaverParams *dest, const std::string &path )
{
	impl->load( dest, path );
}

void IOCtl::Impl::save( const MISC::SaverParams *source, const std::string &path )
{
	s.save( source, path );
}

void IOCtl::Impl::load( MISC::SaverParams *dest, const std::string &path )
{
	s.load( dest, path );
}
