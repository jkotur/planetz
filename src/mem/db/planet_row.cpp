#include "planet_row.h"
#include "rowutils.h"

using namespace MEM;

PlanetRow::PlanetRow()
{
}

PlanetRow::~PlanetRow()
{
}

uint8_t PlanetRow::size() const
{
	return 9;
}

void PlanetRow::setCell( unsigned idx, const std::string& val )
{
	ROW_SWITCH_BEGIN( idx, val )
		ROW_CASE( 0, xcoord )
		ROW_CASE( 1, ycoord )
		ROW_CASE( 2, zcoord )
		ROW_CASE( 3, radius )
		ROW_CASE( 4, mass )
		ROW_CASE( 5, xvel )
		ROW_CASE( 6, yvel )
		ROW_CASE( 7, zvel )
		ROW_CASE( 8, model_id )
	ROW_SWITCH_END()
}

std::string PlanetRow::getTableName() const
{
	return "planets";
}

std::string PlanetRow::getCellNames() const
{
	return "xcoord, ycoord, zcoord, radius, mass, xvel, yvel, zvel, model_id";
}

std::string PlanetRow::getCellDefs() const
{
	return "xcoord REAL, ycoord REAL, zcoord REAL, radius REAL, mass REAL, xvel REAL, yvel REAL, zvel REAL, model_id INT";
}

std::string PlanetRow::getCellValues() const
{
	ROW_VALUES_INIT;
	ROW_VALUES_ADD( xcoord );
	ROW_VALUES_ADD( ycoord );
	ROW_VALUES_ADD( zcoord );
	ROW_VALUES_ADD( radius );
	ROW_VALUES_ADD( mass );
	ROW_VALUES_ADD( xvel );
	ROW_VALUES_ADD( yvel );
	ROW_VALUES_ADD( zvel );
	ROW_VALUES_ADD( model_id );
	return ROW_VALUES_RESULT;
}
