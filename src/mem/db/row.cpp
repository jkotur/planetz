#include "row.h"
#include <sstream>

using namespace MEM;

Row::Row()
{
}

Row::~Row()
{
}

std::string Row::getSaveString() const
{
	std::stringstream ss;
	ss << "INSERT INTO "
		<< getTableName()
		<< "("
		<< getCellNames()
		<< ") VALUES("
		<< getCellValues()
		<< ");";
	return ss.str();
}

std::string Row::getLoadString() const
{
	std::stringstream ss;
	ss << "SELECT "
		<< getCellNames()
		<< " FROM "
		<< getTableName()
		<< ";";
	return ss.str();
}

std::string Row::getCreationString() const
{
	std::stringstream ss;
	ss << "DROP TABLE IF EXISTS "
		<< getTableName()
		<< "; CREATE TABLE "
		<< getTableName()
		<< "("
		<< getCellDefs()
		<< ")";
	return ss.str();
}
