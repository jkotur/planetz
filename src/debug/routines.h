#ifndef _DEBUG_ROUTINES_H_
#define _DEBUG_ROUTINES_H_

#ifndef _RELEASE
#include "../util/logger.h"
#include <assert.h>
#define ASSERT(x) assert(x)
#define ASSERT_MSG(x, y) if(!(x)){log_printf(_ERROR, "%s\n", y); assert(x);}
#define NOENTRY() assert(false)
#define TODO(x) log_printf(_WARNING, "[TODO] %s\n", (x));
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

#endif // _DEBUG_ROUTINES_H_
