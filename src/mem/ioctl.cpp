#include "ioctl.h"

#include <SDL/SDL_image.h>

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
		void loadTextures( MISC::Textures* dest , const std::list<std::string>& names );

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

void IOCtl::loadTextures( MISC::Textures* dest , const std::list<std::string>& names )
{
	impl->loadTextures( dest , names );
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

	              // r    g    b    ke    ka   kd    ks  alpha texId atmR atmG atmB atmDensity atmosphereRadius
	mgr.addMaterial( 1. , 1. , 1. ,  .0 ,  .1 , 5   , 0 , 1 , 0 , .5 , .9 , .2 , 1. , 1.10);
	mgr.addMaterial( 1. , 1. , 1. ,  .0 ,  .1 , 5   , 0 , 1 , 4 , .9 , .5 , .3  , 0 , 1.05 );
	mgr.addMaterial( 1. , 1. , 1. ,  .0 ,  .1 , 5   , 0 , 1 , 2 , 0  , 0  , 0  , 0  ,  .0 );
	mgr.addMaterial( 1. , 1. , 1. ,  .5 , 1.2 , 1.0 , 0 , 1 , 3 , 0  , 0  , 0  , 0  ,  .0 );
	mgr.addMaterial( 1. , 1. , 1. ,  .0 ,  .1 , 5   , 0 , 1 , 1 , 0  , 0  , 0  , 0  ,  .0 );
}

void IOCtl::Impl::loadTextures( MISC::Textures* dest , const std::list<std::string>& names )
{
	for( std::list<std::string>::const_iterator i = names.begin() ; i != names.end() ; ++i )
	{
		SDL_Surface* surface = IMG_Load( i->c_str() );
		if( !surface ) {
			log_printf(_ERROR,"SDL could not load image '%s': %s\n",i->c_str() , SDL_GetError() );
			continue;
		}

		dest->push_back(surface);
	}
}

