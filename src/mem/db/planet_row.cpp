#include "planet_row.h"
#include <cstdio>
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
