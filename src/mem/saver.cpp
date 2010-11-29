#include <misc/phx_planet_factory.h>
#include <misc/gfx_planet_factory.h>

#include "saver.h"
#include "constants.h"
#include "./util/logger.h"
#include "db/dbsqlite.h"
#include "db/table.h"
#include "db/planet_row.h"
#include "db/camera_row.h"
#include "misc/saver_params.h"
#include "ui/camera.h"

#define foreach BOOST_FOREACH

using namespace MEM;
using namespace MEM::MISC;

Saver::Saver()
{
}

Saver::~Saver()
{
	log_printf(DBG,"[DEL] Deleting saver\n");
}

void Saver::save( const MISC::SaverParams *source, const std::string &path )
{
	DBSqlite db( path );
	Table<PlanetRow> table;
	MISC::CpuPlanetHolder *planets = source->planet_info;

	uint32_t num = planets->count[0];

	for( uint32_t i = 0; i < num ; ++i )
	{
		PlanetRow *p = new PlanetRow;
		float3 v = planets->pos[ i ];
		p->xcoord = v.x;
		p->ycoord = v.y;
		p->zcoord = v.z;
		v = planets->velocity[ i ];
		p->xvel = v.x;
		p->yvel = v.y;
		p->zvel = v.z;
		p->mass = planets->mass[ i ];
		p->radius = planets->radius[ i ];
		p->model_id = planets->model[ i ];
		table.add( p );
	}
	db.save( table );

	if( source->cam_info )
	{
		Table<CameraRow> cTable;
		CameraRow *c = new CameraRow;
		c->xcoord = source->cam_info->get_pos().x; 
		c->ycoord = source->cam_info->get_pos().y; 
		c->zcoord = source->cam_info->get_pos().z;
		c->xlook = source->cam_info->get_lookat().x;
		c->ylook = source->cam_info->get_lookat().y;
		c->zlook = source->cam_info->get_lookat().z;
		c->xup = source->cam_info->get_up().x;
		c->yup = source->cam_info->get_up().y;
		c->zup = source->cam_info->get_up().z;
		cTable.add( c );
		db.save( cTable );
	}
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

void Saver::load( MISC::SaverParams *dest, const std::string &path )
{
	MISC::CpuPlanetHolder *h = new MISC::CpuPlanetHolder(); // to można przenieść do SaverParams'ów
	DBSqlite db( path );
	Table<PlanetRow> table;
	db.load( table );

	unsigned i = 0;

	h->resize( table.size() );

	foreach( PlanetRow *p, table )
	{
		h->pos[i] = make_float3( p->xcoord, p->ycoord, p->zcoord );
		h->mass[i] = p->mass;
		h->radius[i] = p->radius;
		h->velocity[i] = make_float3( p->xvel, p->yvel, p->zvel );
		h->model[i] = p->model_id;
		++i;
	}
	dest->planet_info = h;

	Table<CameraRow> cTable;
	db.load( cTable );

	if( cTable.size() && dest->cam_info )
	{
		CameraRow *r = *cTable.begin();
		dest->cam_info->set_perspective(
			Vector3(r->xcoord, r->ycoord, r->zcoord),
			Vector3(r->xlook, r->ylook, r->zlook),
			Vector3(r->xup, r->yup, r->zup) );
	}
}
