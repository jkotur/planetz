#include <misc/phx_planet_factory.h>
#include <misc/gfx_planet_factory.h>

#include "saver.h"
#include "constants.h"
#include "./util/logger.h"
#include "db/dbsqlite.h"
#include "db/table.h"
#include "db/planet_row.h"

using namespace MEM;
using namespace MEM::MISC;

Saver::Saver()
{
}

Saver::~Saver()
{
	log_printf(DBG,"[DEL] Deleting saver\n");
}

void Saver::save( PhxPlanetFactory* planets_phx, GfxPlanetFactory* planets_gfx, const std::string &path )
{
	log_printf(DBG, "Creating db object\n");
	DBSqlite db( path );
	log_printf(DBG, "Creating table object\n");
	Table<PlanetRow> table;

	uint32_t num = planets_phx->getCount().map( BUF_H )[0];

	map_buffers( planets_phx, planets_gfx );

	for( uint32_t i = 0; i < num ; ++i )
	{
		PlanetRow *p = new PlanetRow;

		float3 v = planets_phx->getPositions().map( BUF_H )[ i ];
		p->xcoord = v.x;
		p->ycoord = v.y;
		p->zcoord = v.z;
		p->xvel = v.x;
		p->yvel = v.y;
		p->zvel = v.z;
		p->mass = planets_phx->getMasses().h_data()[ i ];
		p->radius = planets_phx->getRadiuses().map( BUF_H )[ i ];
		log_printf(DBG, "Adding to table\n");
		table.add( p );
	}

	unmap_buffers( planets_phx, planets_gfx );

	log_printf(DBG, "saving to db\n");
	db.save( table );
	log_printf(DBG, "done\n");
}

void Saver::load( PhxPlanetFactory* planets_phx, GfxPlanetFactory* planets_gfx, const std::string &path )
{
	DBSqlite db( path );
	Table<PlanetRow> table;
	db.load( table );

	float3 pos;
	float3 speed;
	double mass;
	double radius;

	map_buffers( planets_phx, planets_gfx );


	BOOST_FOREACH( PlanetRow *p, table )
	{
		pos.x = p->xcoord;
		pos.y = p->ycoord;
		
		pos.z = p->zcoord;
		speed.x = p->xvel;
		speed.y = p->yvel;
		speed.z = p->zvel;
		mass = p->mass;
		radius = p->radius;

		log_printf(DBG,"[LOAD] Adding planet at (%f,%f,%f) with speed (%f,%f,%f), mass %f and radius %f\n"
			,pos.x,pos.y,pos.z
			,speed.x,speed.y,speed.z
			,mass,radius );
	}

	unmap_buffers( planets_phx, planets_gfx );
}

void Saver::map_buffers( PhxPlanetFactory *planets_phx, GfxPlanetFactory *planets_gfx )
{
	planets_phx->getRadiuses().map( BUF_H );
	planets_phx->getPositions().map( BUF_H );
	planets_phx->getMasses().bind();
}

void Saver::unmap_buffers( PhxPlanetFactory *planets_phx, GfxPlanetFactory *planets_gfx )
{
	planets_phx->getRadiuses().unmap();
	planets_phx->getPositions().unmap();
	planets_phx->getMasses().unbind();
}
