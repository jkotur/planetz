#include "textures_row.h"
#include "rowutils.h"

using namespace MEM;

TexturesRow::TexturesRow()
{
}

TexturesRow::~TexturesRow()
{
}

uint8_t TexturesRow::size() const
{
	return 2;
}

void TexturesRow::setCell( unsigned idx, const std::string &val )
{
	ROW_SWITCH_BEGIN( idx, val )
		ROW_CASE( 0, id )
		ROW_CASE( 1, path )
	ROW_SWITCH_END()
}

std::string TexturesRow::getTableName() const
{
	return "textures";
}

std::string TexturesRow::getCellNames() const
{
	return "id, path";
}

std::string TexturesRow::getCellDefs() const
{
	return "id INT, path TEXT";
}

std::string TexturesRow::getCellValues() const
{
	ROW_VALUES_INIT;
	ROW_VALUES_ADD( id );
	ROW_VALUES_ADD( path );
	return ROW_VALUES_RESULT;
}
