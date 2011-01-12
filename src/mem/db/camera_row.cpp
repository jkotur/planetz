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

std::string CameraRow::getSaveString() const
{
	std::stringstream ss1, ss2;
	for( unsigned i = 0; i < 16; ++i )
	{
		if( i )
		{
			ss1 << ", ";
			ss2 << ", ";
		}
		ss1 << param(i);
		ss2 << matrix[i];
	}
	std::stringstream retval;
	retval 
		<< "INSERT INTO camera("
		<< ss1.str()
		<< ") VALUES("
		<< ss2.str()
		<< ");";
	return retval.str();
}

std::string CameraRow::getLoadString() const
{
	std::stringstream ss;
	ss << "SELECT ";
	for( unsigned i = 0; i < 16; ++i )
	{
		if( i )
		{
			ss << ", ";
		}
		ss << param(i);
	}
	ss << " FROM camera;";
	return ss.str();
}

std::string CameraRow::getCreationString() const
{
	std::stringstream ss;
	ss << "DROP TABLE IF EXISTS camera; CREATE TABLE camera(";
	for( unsigned i = 0; i < 16; ++i )
	{
		if( i )
		{
			ss << ", ";
		}
		ss << param(i) << " REAL";
	}
	ss << ");";
	return ss.str();
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

