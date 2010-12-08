#include "ioctl.h"
#include "saver.h"
#include "constants.h"

#include "debug/routines.h"

using namespace MEM;

class IOCtl::Impl
{
	public:
		void save( const MISC::SaverParams *source, const std::string& path );
		void load( MISC::SaverParams *dest, const std::string& path );

		void loadMaterials( MISC::Materials* dest , const std::string & path );

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

void IOCtl::loadMaterials( MISC::Materials* dest , const std::string & path )
{
	impl->loadMaterials( dest , path );
}

void IOCtl::Impl::save( const MISC::SaverParams *source, const std::string &path )
{
	s.save( source, path );
}

void IOCtl::Impl::load( MISC::SaverParams *dest, const std::string &path )
{
	s.load( dest, path );
}

void IOCtl::Impl::loadMaterials( MISC::Materials* dest , const std::string & path )
{
	TODO("Load materials from file or sth");

	MISC::MaterialsMgr mgr( dest );

	               // r    g    b   ke   ka   kd    ks  alpha
	mgr.addMaterial( .5 , .1 , .0 , .0 , .3 , 10 , 0 , 1 );
	mgr.addMaterial( .0 , .3 , 1. , .0 , .3 , 10 , 0 , 1 );
	mgr.addMaterial( .5 , 1. , .0 , .0 , .3 , 10 , 0 , 1 );
	mgr.addMaterial( 1. , 1. , 1. , .1 , .3 , 1.0 , 0 , 1 );
}

