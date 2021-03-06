#include <boost/filesystem.hpp>
#include "util/logger.h"
#include "data_flow_mgr.h"
#include "constants.h"
#include "ioctl.h"
#include "memory_manager.h"
#include "misc/saver_params.h"
#include "cuda/math.h"

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

		void registerCam( UI::CameraMgr *_cam );
		unsigned createPlanet( MISC::PlanetParams params );
		void removePlanet( unsigned id );
		void dropPlanets();

	private:
		void updateBuffers( MISC::CpuPlanetHolder*p, MISC::Materials*m );

		IOCtl ioctl;
		MemMgr memmgr;
		UI::CameraMgr *cam;
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
	if( !boost::filesystem::is_regular_file( path ) ) {
		log_printf(_WARNING,"File %s does not exist\n",path.c_str());
		return;
	}
	MISC::SaverParams p( cam );
	ioctl.load( &p, path );
	log_printf(DBG, "got %u planets\n", p.planet_info->size() );
	updateBuffers( p.planet_info , materials );
	if( p.planet_info->size() <= 0 ) {
		log_printf(_WARNING,"Empty save file: %s\n",path.c_str());
		return;
	}
	memmgr.setPlanets( p.planet_info );
}

void DataFlowMgr::Impl::updateBuffers(MISC::CpuPlanetHolder*p , MISC::Materials*m)
{
	if( !m ) {
		log_printf(_ERROR,"There are no materials loaded\n");
		for( unsigned i=0 ; i<p->size() ; i++ ) {
			p->light    [i]   = make_float3(0);
			p->texId    [i]   = 0; // dummy texture
			p->atm_data [i].y = 0; // no atmosphere
		}
	} else	for( unsigned i=0 ; i<p->size() ; i++ ) {
			p->light    [i]   = make_float3(
					(*m)[p->model[i]].ke ,
					(*m)[p->model[i]].ka ,
					(*m)[p->model[i]].kd );
			p->texId    [i]   = (*m)[p->model[i]].texture;
			p->atm_color[i].x = (*m)[p->model[i]].ar;
			p->atm_color[i].y = (*m)[p->model[i]].ag;
			p->atm_color[i].z = (*m)[p->model[i]].ab;
			p->atm_data [i].x = (*m)[p->model[i]].ad;
			p->atm_data [i].y = (*m)[p->model[i]].al;
		}
}

GLuint DataFlowMgr::Impl::loadTextures()
{
	TODO("Read texture files from file");

	MISC::Textures tex;
	ioctl.loadTextures(&tex,DATA("textures/index.db"));
	GLuint res = memmgr.loadTextures( tex );
	for( MISC::Textures::iterator i = tex.begin() ; i != tex.end() ; ++i )
		SDL_FreeSurface( i->second );
	return res;
}

GLuint DataFlowMgr::Impl::loadMaterials()
{
	if( materials ) delete materials;
	materials = new MISC::Materials();
	ioctl.loadMaterials( materials , DATA("materials.db") );
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

void DataFlowMgr::Impl::registerCam( UI::CameraMgr *_cam )
{
	cam = _cam;
}

unsigned DataFlowMgr::Impl::createPlanet( MISC::PlanetParams params )
{
	unsigned id = memmgr.createPlanet( params );
	unsigned mid = params.model;
	MISC::GfxPlanetFactory *f = getGfxMem();
	if( materials )
	{
		const_cast<MISC::BufferGl<float3>& >( f->getLight() )
			.setAt( id, make_float3(
					(*materials)[mid].ke ,
					(*materials)[mid].ka ,
					(*materials)[mid].kd ) );
		const_cast<MISC::BufferGl<int>& >( f->getTexIds() )
			.setAt( id, (*materials)[mid].texture );
		const_cast<MISC::BufferGl<float3>& >( f->getAtmColor() )
			.setAt( id, make_float3(
					(*materials)[mid].ar,
					(*materials)[mid].ag,
					(*materials)[mid].ab ) );
		const_cast<MISC::BufferGl<float2>& >( f->getAtmData() )
			.setAt( id, make_float2(
					(*materials)[mid].ad,
					(*materials)[mid].al ) );
	}
	else
	{
		const_cast<MISC::BufferGl<float3>& >( f->getLight() )
			.setAt( id, make_float3(0) );
		const_cast<MISC::BufferGl<int>& >( f->getTexIds() )
			.setAt( id, 0 );
		const_cast<MISC::BufferGl<float2>& >( f->getAtmData() )
			.setAt( id, make_float2(0) );
	}
	return id;
}

void DataFlowMgr::Impl::dropPlanets()
{
	memmgr.dropPlanets();
}

void DataFlowMgr::Impl::removePlanet( unsigned id )
{
	memmgr.removePlanet( id );
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

void DataFlowMgr::registerCam( UI::CameraMgr *cam )
{
	impl->registerCam( cam );
}

unsigned DataFlowMgr::createPlanet( MISC::PlanetParams params )
{
	return impl->createPlanet( params );
}

void DataFlowMgr::removePlanet( unsigned id )
{
	impl->removePlanet( id );
}

void DataFlowMgr::dropPlanets()
{
	impl->dropPlanets();
}
