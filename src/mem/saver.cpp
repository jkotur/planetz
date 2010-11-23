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

void Saver::save( const MISC::CpuPlanetHolder *source, const std::string &path )
{
	DBSqlite db( path );
	Table<PlanetRow> table;

	uint32_t num = source->count[0];

	for( uint32_t i = 0; i < num ; ++i )
	{
		PlanetRow *p = new PlanetRow;
		float3 v = source->pos[ i ];
		p->xcoord = v.x;
		p->ycoord = v.y;
		p->zcoord = v.z;
		v = source->velocity[ i ];
		p->xvel = v.x;
		p->yvel = v.y;
		p->zvel = v.z;
		p->mass = source->mass[ i ];
		p->radius = source->radius[ i ];
		p->model_id = source->model[ i ];
		table.add( p );
	}
	db.save( table );
}

namespace
{
	float3 make_float3( float x, float y, float z )
	{
		float3 f;
		f.x = x;
		f.y = y;
		f.z = z;
		return f;
	}
}

MISC::CpuPlanetHolder* Saver::load( const std::string &path )
{
	TODO("use some kind of smart ptr instead of returning allocated raw ptr");
	MISC::CpuPlanetHolder *h = new MISC::CpuPlanetHolder();
	DBSqlite db( path );
	Table<PlanetRow> table;
	db.load( table );

	unsigned i = 0;

	h->resize( table.size() );

	BOOST_FOREACH( PlanetRow *p, table )
	{
		h->pos[i] = make_float3( p->xcoord, p->ycoord, p->zcoord );
		h->mass[i] = p->mass;
		h->radius[i] = p->radius;
		h->velocity[i] = make_float3( p->xvel, p->yvel, p->zvel );
		h->model[i] = p->model_id;
		++i;
	}
	return h;
}
