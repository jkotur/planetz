#ifndef _DEBUG_ROUTINES_H_
#define _DEBUG_ROUTINES_H_

#define WHERESTR  "[file %s, line %d] "
#define WHEREARG  __FILE__, __LINE__

#ifndef _RELEASE
#include "../util/logger.h"
#include <assert.h>
#include <stdlib.h>
#define ASSERT(x) assert(x)
#define ASSERT_MSG_2(...) log_printf(__VA_ARGS__);
#define ASSERT_MSG(x, form , ... ) do{if(!(x)){ASSERT_MSG_2(_ERROR,"[ASSERT]" WHERESTR form "\n" , WHEREARG , ##__VA_ARGS__ ); assert(x);}}while(0);
#define NOENTRY() do{log_printf(_ERROR, "This line should never be reached: '%s:%d'\n", __FILE__,__LINE__); abort();}while(0);
#define TODO(x) log_printf(_WARNING, "[TODO]" WHERESTR " %s\n", WHEREARG , x);
#define DBGPUT(x) x
#define RELPUT(x)

#else
#define ASSERT(x)
#define ASSERT_MSG(x, y)
#define NOENTRY()
#define TODO(x)
#define DBGPUT(x)
#define RELPUT(x) x

#endif

#if 0
// There can go code for disabling boost warnings. Unfortunetely, gcc <4.6 seems not to be able to turn off warnings for just some include files. And since I don't have gcc4.6 at this moment, this is not fully implemented.
#define DO_PRAGMA(x) _Pragma (#x)

#define WARNING_SWITCH(action) \
DO_PRAGMA (GCC diagnostic action "-Wstrict-aliasing")\
DO_PRAGMA (GCC diagnostic action "-Wunused-variable")

#define DISABLE_WARNINGS_BEGIN \
WARNING_SWITCH( ignored )

#define DISABLE_WARNINGS_END \
WARNING_SWITCH( warning )

#endif

#endif // _DEBUG_ROUTINES_H_
