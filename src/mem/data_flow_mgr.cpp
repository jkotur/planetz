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

		MISC::GfxPlanetFactory *getGfxMem();
		MISC::PhxPlanetFactory *getPhxMem();

		void registerCam( Camera *_cam );

	private:
		IOCtl ioctl;
		MemMgr memmgr;
		Camera *cam;
};

DataFlowMgr::Impl::Impl()
	: cam( NULL )
{
}

DataFlowMgr::Impl::~Impl()
{
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
	memmgr.setPlanets( p.planet_info );
}

GLuint DataFlowMgr::Impl::loadMaterials()
{
	return memmgr.loadMaterials();
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

void DataFlowMgr::registerCam( Camera *cam )
{
	impl->registerCam( cam );
}
