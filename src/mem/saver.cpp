#include <fstream>

#include "saver.h"
#include "constants.h"
#include "./util/logger.h"

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

void Saver::save( const std::string& path )
{
	std::fstream file( path.c_str() ,  std::ios_base::out | std::ios_base::trunc );

	Vector3 v = cam.get_pos();
	file << v.x << " " << v.y  << " "<< v.z << std::endl;
	v = cam.get_lookat();
	file << v.x  << " "<< v.y  << " "<< v.z << std::endl;
	v = cam.get_up();
	file << v.x  << " "<< v.y  << " "<< v.z << std::endl;
	for( Planetz::iterator i = plz.begin() ; i != plz.end() ; ++i )
	{
		v = (*i)->get_phx()->get_pos();
		file << v.x  << " "<< v.y  << " "<< v.z << " ";
		v = (*i)->get_phx()->get_velocity();
		file << v.x  << " "<< v.y  << " "<< v.z << " ";
		file << (*i)->get_phx()->get_mass() << " ";
		file << (*i)->get_phx()->get_radius() << std::endl;
	}
	file.close();
}

void Saver::load( const std::string& path )
{
	std::fstream file( path.c_str() ,  std::ios_base::in );

	Vector3 pos;
	file >> pos.x >> pos.y >> pos.z;
	Vector3 lookat;
	file >> lookat.x >> lookat.y >> lookat.z;
	Vector3 up;
	file >> up.x >> up.y >> up.z;

	cam.set_perspective(pos,lookat,up);

	plz.clear();

	Vector3 speed;
	double mass;
	double radius;
	while( file.good() )
	{
		file >> pos.x >> pos.y >> pos.z;
		file >> speed.x >> speed.y >> speed.z;
		file >> mass;
		file >> radius;
		
		if( file.eof() ) break;

	log_printf(DBG,"[LOAD] Adding planet at (%f,%f,%f) with speed (%f,%f,%f), mass %f and radius %f\n"
			,pos.x,pos.y,pos.z
			,speed.x,speed.y,speed.z
			,mass,radius );

		GFX::Planet*gp = new GFX::Planet( );
		Phx::Planet*pp = new Phx::Planet( pos , speed , mass , radius );
		plz.add( new Planet(gp,pp) );
	}
}

