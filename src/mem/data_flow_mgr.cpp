#include "util/logger.h"
#include "data_flow_mgr.h"
#include "constants.h"
#include "ioctl.h"
#include "memory_manager.h"
#include "misc/saver_params.h"

using namespace MEM;

static const char *DEFAULT_SAVE_FILE = "saves/qsave.sav";

class DataFlowMgr::Impl
{
	public:
		Impl();
		~Impl();

		void save( const std::string &path = DATA( DEFAULT_SAVE_FILE ) );
		void load( const std::string &path = DATA( DEFAULT_SAVE_FILE ) );

		GLuint loadMaterials();
		GLuint loadTextures();

		MISC::GfxPlanetFactory *getGfxMem();
		MISC::PhxPlanetFactory *getPhxMem();

		void registerCam( Camera *_cam );

	private:
		void updateEmissive( MISC::CpuPlanetHolder*p, MISC::Materials*m );

		IOCtl ioctl;
		MemMgr memmgr;
		Camera *cam;
		MISC::Materials*materials;
};

DataFlowMgr::Impl::Impl()
	: cam( NULL ) , materials(NULL)
{
}

DataFlowMgr::Impl::~Impl()
{
	if( materials ) delete materials;
}

void DataFlowMgr::Impl::save( const std::string &path )
{
	MISC::CpuPlanetHolder *planets = memmgr.getPlanets();
	MISC::SaverParams p( cam );
	p.planet_info = planets;
	ioctl.save( &p, path );
	log_printf(DBG, "saved planets\n");
}

void DataFlowMgr::Impl::load( const std::string &path )
{
	MISC::SaverParams p( cam );
	ioctl.load( &p, path );
	log_printf(DBG, "got %u planets\n", p.planet_info->size() );
	updateEmissive( p.planet_info , materials );
	memmgr.setPlanets( p.planet_info );
}

void DataFlowMgr::Impl::updateEmissive(MISC::CpuPlanetHolder*p , MISC::Materials*m)
{
	if( !m ) {
		log_printf(_WARNING,"There are no materials loaded\n");
		for( unsigned i=0 ; i<p->size() ; i++ )
			p->emissive[i] = 0;
	} else	for( unsigned i=0 ; i<p->size() ; i++ )
			p->emissive[i] = (*m)[p->model[i]].ke;
}

GLuint DataFlowMgr::Impl::loadTextures()
{
	MISC::Textures tex;
	std::list<std::string> names;
	names.push_back( DATA("textures/small_earth.jpg") );
	names.push_back( DATA("textures/small_jupiter.jpg") );
	names.push_back( DATA("textures/small_saturn.jpg") );
	names.push_back( DATA("textures/small_sun.jpg") );
	ioctl.loadTextures(&tex,names);
	GLuint res = memmgr.loadTextures( tex );
	for( MISC::Textures::iterator i = tex.begin() ; i != tex.end() ; ++i )
		SDL_FreeSurface( *i );
	return res;
}

GLuint DataFlowMgr::Impl::loadMaterials()
{
	if( materials ) delete materials;
	materials = new MISC::Materials();
	ioctl.loadMaterials( materials , "should be here?" );
	return memmgr.loadMaterials(*materials );
}

MISC::GfxPlanetFactory *DataFlowMgr::Impl::getGfxMem()
{
	return memmgr.getGfxMem();
}

MISC::PhxPlanetFactory *DataFlowMgr::Impl::getPhxMem()
{
	return memmgr.getPhxMem();
}

void DataFlowMgr::Impl::registerCam( Camera *_cam )
{
	cam = _cam;
}

DataFlowMgr::DataFlowMgr()
	: impl( new DataFlowMgr::Impl() )
{
}

DataFlowMgr::~DataFlowMgr()
{
	delete impl;
}

void DataFlowMgr::save( const std::string &path )
{
	impl->save( path );
}

void DataFlowMgr::load( const std::string &path )
{
	impl->load( path );
}

void DataFlowMgr::save()
{
	impl->save();
}

void DataFlowMgr::load()
{
	impl->load();
}

MISC::GfxPlanetFactory *DataFlowMgr::getGfxMem()
{
	return impl->getGfxMem();
}

MISC::PhxPlanetFactory *DataFlowMgr::getPhxMem()
{
	return impl->getPhxMem();
}

GLuint DataFlowMgr::loadMaterials()
{
	return impl->loadMaterials();
}

GLuint DataFlowMgr::loadTextures()
{
	return impl->loadTextures();
}

void DataFlowMgr::registerCam( Camera *cam )
{
	impl->registerCam( cam );
}
