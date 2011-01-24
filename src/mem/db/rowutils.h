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

#define ROW_VALUES_INIT \
	std::stringstream ss

#define ROW_VALUES_ADD(x) \
	ss << (x) << ", "

#define ROW_VALUES_RESULT \
	ss.str().substr( 0, ss.str().size() - 2 )

#endif // _DB_ROWUTILS_H_
