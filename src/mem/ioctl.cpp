#include "ioctl.h"

#include <SDL/SDL_image.h>

#include "saver.h"
#include "constants.h"

#include "debug/routines.h"
#include "util/logger.h"

#include "db/dbsqlite.h"
#include "db/table.h"
#include "db/materials_row.h"
#include "db/textures_row.h"

#define foreach BOOST_FOREACH

using namespace MEM;

class IOCtl::Impl
{
	public:
		void save( const MISC::SaverParams *source, const std::string& path );
		void load( MISC::SaverParams *dest, const std::string& path );

		void loadMaterials( MISC::Materials* dest , const std::string & path );
		void loadTextures( MISC::Textures* dest , const std::string & path );

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

void IOCtl::loadTextures( MISC::Textures* dest , const std::string & path )
{
	impl->loadTextures( dest , path );
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

	DBSqlite db( path );
	Table<MaterialsRow> table;
	db.load( table );

	foreach( MaterialsRow *m, table )
	{
		mgr.addMaterial( m->col.r, m->col.g, m->col.b, m->ke, m->ka, m->kd,
			0/*ks*/, m->alpha, m->texId, m->atm.r, m->atm.g, m->atm.b,
			m->atmDensity, m->atmRadius );
	}
	              //   r    g    b    ke    ka   kd    ks  alpha texId atmR atmG atmB atmDensity atmosphereRadius
	//mgr.addMaterial( 1. , 1. , 1. ,  .0 ,  .0 , 5   , 0 , 1 , 0 , .5 , .9 , .2 , 1. , 1.10 );
}

void IOCtl::Impl::loadTextures( MISC::Textures* dest , const std::string & path )
{
	DBSqlite db( path );
	Table<TexturesRow> table;
	db.load( table );

	foreach( TexturesRow* t, table )
	{
		std::string absolute_path = DATA( "textures/" ) + t->path;
		SDL_Surface* surface = IMG_Load( absolute_path.c_str() );
		if( !surface ) {
			log_printf(_ERROR,"SDL could not load image '%s': %s\n",absolute_path.c_str() , SDL_GetError() );
			continue;
		}

		(*dest)[ t->id ] = surface;
	}
}

