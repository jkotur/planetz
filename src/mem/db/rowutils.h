#ifndef _DB_ROWUTILS_H_
#define _DB_ROWUTILS_H_

#include <cstdio>
#include <sstream>
#include <debug/routines.h>

#define ROW_SWITCH_BEGIN( idx, x )  \
	std::stringstream ss( x );  \
	switch( idx )               \
	{
#define ROW_CASE( x, var )\
	case x:           \
		ss >> var;\
		break;
#define ROW_SWITCH_END()  \
	default:          \
		NOENTRY();\
	}

#endif // _DB_ROWUTILS_H_
