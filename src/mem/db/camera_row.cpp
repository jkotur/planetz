#include "camera_row.h"
#include "rowutils.h"

using namespace MEM;

const std::string CameraRow::save_string = "INSERT INTO camera(xcoord, ycoord, zcoord, xlook, ylook, zlook, xup, yup, zup) VALUES(%f, %f, %f, %f, %f, %f, %f, %f, %f);";
const std::string CameraRow::load_string = "SELECT xcoord, ycoord, zcoord, xlook, ylook, zlook, xup, yup, zup FROM camera;";
const std::string CameraRow::creation_string = "DROP TABLE IF EXISTS camera; CREATE TABLE camera(xcoord REAL, ycoord REAL, zcoord REAL, xlook REAL, ylook REAL, zlook REAL, xup REAL, yup REAL, zup REAL);";

CameraRow::CameraRow()
{
}

CameraRow::~CameraRow()
{
}

std::string CameraRow::getSaveString() const
{
	TODO("make it more safely, too");
	char *buf = new char[ save_string.size() + 666 ];
	sprintf(buf, save_string.c_str(), xcoord, ycoord, zcoord, xlook, ylook, zlook, xup, yup, zup);
	std::string retval( buf );
	delete[] buf;
	return retval;
}

std::string CameraRow::getLoadString() const
{
	return load_string;
}

std::string CameraRow::getCreationString() const
{
	return creation_string;
}

uint8_t CameraRow::size() const
{
	return 9;
}

void CameraRow::setCell( unsigned idx, const std::string &val )
{
	ROW_SWITCH_BEGIN( idx, val )
		ROW_CASE( 0, xcoord )
		ROW_CASE( 1, ycoord )
		ROW_CASE( 2, zcoord )
		ROW_CASE( 3, xlook )
		ROW_CASE( 4, ylook )
		ROW_CASE( 5, zlook )
		ROW_CASE( 6, xup )
		ROW_CASE( 7, yup )
		ROW_CASE( 8, zup )
	ROW_SWITCH_END()
}

