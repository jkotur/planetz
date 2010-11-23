#include "planet_row.h"
#include <cstdio>
#include <sstream>
#include <debug/routines.h>

using namespace MEM;

const std::string PlanetRow::save_string = "INSERT INTO planets(xcoord, ycoord, zcoord, radius, mass, xvel, yvel, zvel, model_id) VALUES(%f, %f, %f, %f, %f, %f, %f, %f, %u);";
const std::string PlanetRow::load_string = "SELECT xcoord, ycoord, zcoord, radius, mass, xvel, yvel, zvel, model_id FROM planets;";
const std::string PlanetRow::creation_string = "CREATE TABLE planets(xcoord REAL, ycoord REAL, zcoord REAL, radius REAL, mass REAL, xvel REAL, yvel REAL, zvel REAL, model_id INT);";

PlanetRow::PlanetRow()
{
}

PlanetRow::~PlanetRow()
{
}

std::string PlanetRow::getSaveString() const
{
	TODO("make it more safely");
	char *buf = new char[ save_string.size() + 666 ];
	sprintf(buf, save_string.c_str(), xcoord, ycoord, zcoord, radius, mass, xvel, yvel, zvel, model_id);
	std::string retval( buf );
	delete buf;
	return retval;
}

std::string PlanetRow::getLoadString() const
{
	return load_string;
}

std::string PlanetRow::getCreationString() const
{
	return creation_string;
}

uint8_t PlanetRow::size() const
{
	return 9;
}

void PlanetRow::setCell( unsigned idx, const std::string& val )
{
	std::stringstream ss( val );
	switch( idx )
	{
	case 0:
		ss >> xcoord;
		break;
	case 1:
		ss >> ycoord;
		break;
	case 2:
		ss >> zcoord;
		break;
	case 3:
		ss >> radius;
		break;
	case 4:
		ss >> mass;
		break;
	case 5:
		ss >> xvel;
		break;
	case 6:
		ss >> yvel;
		break;
	case 7:
		ss >> zvel;
		break;
	case 8:
		ss >> model_id;
		break;
	default:
		NOENTRY();
	}
}
