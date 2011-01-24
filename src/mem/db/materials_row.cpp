#include "materials_row.h"
#include "rowutils.h"

using namespace MEM;

MaterialsRow::MaterialsRow()
{
}

MaterialsRow::~MaterialsRow()
{
}

uint8_t MaterialsRow::size() const
{
	return 14;
}

void MaterialsRow::setCell( unsigned idx, const std::string &val )
{
	ROW_SWITCH_BEGIN( idx, val )
		ROW_CASE( 0 , id );
		ROW_CASE( 1 , col.r );
		ROW_CASE( 2 , col.g );
		ROW_CASE( 3 , col.b );
		ROW_CASE( 4 , ke );
		ROW_CASE( 5 , ka );
		ROW_CASE( 6 , kd );
		ROW_CASE( 7 , alpha );
		ROW_CASE( 8 , atm.r );
		ROW_CASE( 9 , atm.g );
		ROW_CASE( 10, atm.b );
		ROW_CASE( 11, atmDensity );
		ROW_CASE( 12, atmRadius );
		ROW_CASE( 13, texId );
	ROW_SWITCH_END()
}

std::string MaterialsRow::getTableName() const
{
	return "material";
}

std::string MaterialsRow::getCellNames() const
{
	return "id, colr, colg, colb, ke, ka, kd, alpha, atmr, atmg, atmb, atmDensity, atmRadius, texId";
}

std::string MaterialsRow::getCellDefs() const
{
	return "id INT, colr REAL, colg REAL, colb REAL, ke REAL, ka REAL, kd REAL, alpha REAL, atmr REAL, atmg REAL, atmb REAL, atmDensity REAL, atmRadius REAL, texId INT";
}

std::string MaterialsRow::getCellValues() const
{
	ROW_VALUES_INIT;
	ROW_VALUES_ADD( id );
	ROW_VALUES_ADD( col.r );
	ROW_VALUES_ADD( col.g );
	ROW_VALUES_ADD( col.b );
	ROW_VALUES_ADD( ke );
	ROW_VALUES_ADD( ka );
	ROW_VALUES_ADD( kd );
	ROW_VALUES_ADD( alpha );
	ROW_VALUES_ADD( atm.r );
	ROW_VALUES_ADD( atm.g );
	ROW_VALUES_ADD( atm.b );
	ROW_VALUES_ADD( atmDensity );
	ROW_VALUES_ADD( atmRadius );
	ROW_VALUES_ADD( texId );
	return ROW_VALUES_RESULT;
}
