#include "data_flow_mgr.h"
#include "constants.h"
#include "ioctl.h"
#include "memory_manager.h"

using namespace MEM;

static const char *DEFAULT_SAVE_FILE = "saves/first.sav";

class DataFlowMgr::Impl
{
	public:
		Impl();
		~Impl();

		void save( const std::string &path = DATA( DEFAULT_SAVE_FILE ) );
		void load( const std::string &path = DATA( DEFAULT_SAVE_FILE ) );

		MISC::GfxPlanetFactory *getGfxMem();
		MISC::PhxPlanetFactory *getPhxMem();

	private:
		IOCtl ioctl;
		MemMgr memmgr;
};

DataFlowMgr::Impl::Impl()
{
}

DataFlowMgr::Impl::~Impl()
{
}

void DataFlowMgr::Impl::save( const std::string &path )
{
	log_printf(DBG, "savin:\n");
	MISC::CpuPlanetHolder *planets = memmgr.getPlanets();
	ioctl.save( planets, path );
	log_printf(DBG, "saved planets\n");
	delete planets;
}

void DataFlowMgr::Impl::load( const std::string &path )
{
	MISC::CpuPlanetHolder *planets = ioctl.load( path );
	log_printf(DBG, "got %u planets\n", planets->size() );
	for( unsigned i = 0; i < planets->size(); ++i )
	{
		log_printf(DBG, "mass %f\n", planets->mass[i]);
	}
	memmgr.setPlanets( planets );
	delete planets;
}

MISC::GfxPlanetFactory *DataFlowMgr::Impl::getGfxMem()
{
	return memmgr.getGfxMem();
}

MISC::PhxPlanetFactory *DataFlowMgr::Impl::getPhxMem()
{
	return memmgr.getPhxMem();
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
