#include "camera_row.h"
#include "rowutils.h"
#include <sstream>

std::string param( unsigned i )
{
	std::stringstream ss;
	ss << "x" << i;
	return ss.str();
}

using namespace MEM;

CameraRow::CameraRow()
{
}

CameraRow::~CameraRow()
{
}

uint8_t CameraRow::size() const
{
	return 16;
}

void CameraRow::setCell( unsigned idx, const std::string &val )
{
	std::stringstream ss( val );
	ss >> matrix[ idx ];
}

std::string CameraRow::getTableName() const
{
	return "camera";
}

std::string CameraRow::getCellNames() const
{
	std::stringstream ss;
	for( unsigned i = 0; i < 16; ++i )
	{
		if( i )
		{
			ss << ", ";
		}
		ss << param(i);
	}
	return ss.str();
}

std::string CameraRow::getCellDefs() const
{
	std::stringstream ss;
	for( unsigned i = 0; i < 16; ++i )
	{
		if( i )
		{
			ss << ", ";
		}
		ss << param(i) << " REAL";
	}
	return ss.str();
}

std::string CameraRow::getCellValues() const
{
	ROW_VALUES_INIT;
	for( unsigned i = 0; i < 16; ++i )
		ROW_VALUES_ADD( matrix[i] );
	return ROW_VALUES_RESULT;
}
