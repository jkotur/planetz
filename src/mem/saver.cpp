#include <fstream>

#include "saver.h"
#include "constants.h"
#include "./util/logger.h"
#include "db/dbsqlite.h"
#include "db/table.h"
#include "db/planet_row.h"

using namespace MEM;

Saver::Saver( Planetz& _p , Camera& _c )
	: plz(_p) , cam(_c) 
{
}

Saver::~Saver()
{
	log_printf(DBG,"[DEL] Deleting saver\n");
}

void Saver::save()
{
	log_printf(DBG,"saving\n");
	save( DATA("saves/first.sav") );
}

void Saver::load()
{
	load( DATA("saves/first.sav") );
}

void Saver::save( const std::string &path )
{
	log_printf(DBG, "Creating db object\n");
	DBSqlite db( path );
	log_printf(DBG, "Creating table object\n");
	Table<PlanetRow> table;
	Vector3 v;

	for( Planetz::iterator i = plz.begin() ; i != plz.end() ; ++i )
	{
		log_printf(DBG, "Creating PlanetRow object\n");
		PlanetRow *p = new PlanetRow();
		v = (*i)->get_phx()->get_pos();
		p->xcoord = v.x;
		p->ycoord = v.y;
		p->zcoord = v.z;
		v = (*i)->get_phx()->get_velocity();
		p->xvel = v.x;
		p->yvel = v.y;
		p->zvel = v.z;
		p->mass = (*i)->get_phx()->get_mass();
		p->radius = (*i)->get_phx()->get_radius();
		log_printf(DBG, "Adding to table\n");
		table.add( p );
	}

	log_printf(DBG, "saving to db\n");
	db.save( table );
	log_printf(DBG, "done\n");
}

void Saver::load( const std::string &path )
{
	DBSqlite db( path );
	Table<PlanetRow> table;
	db.load( table );
	plz.clear();

	Vector3 pos;
	Vector3 speed;
	double mass;
	double radius;

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

		GFX::Planet*gp = new GFX::Planet( );
		Phx::Planet*pp = new Phx::Planet( pos , speed , mass , radius );
		plz.add( new Planet(gp,pp) );
	}
}

