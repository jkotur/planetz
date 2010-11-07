#ifndef _DEBUG_ROUTINES_H_
#define _DEBUG_ROUTINES_H_

#define WHERESTR  "[file %s, line %d] "
#define WHEREARG  __FILE__, __LINE__

#ifndef _RELEASE
#include "../util/logger.h"
#include <assert.h>
#define ASSERT(x) assert(x)
#define ASSERT_MSG_2(...) log_printf(__VA_ARGS__);
#define ASSERT_MSG(x, form , ... ) do{if(!(x)){ASSERT_MSG_2(_ERROR,WHERESTR "[ASSERT]" form "\n" , WHEREARG , ##__VA_ARGS__ ); assert(x);}}while(0);
#define NOENTRY() assert(false)
#define TODO(x) log_printf(_WARNING, WHERESTR "[TODO] %s\n", WHEREARG , x);
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
