#ifndef __LOGGER_H__

#define __LOGGER_H__

#include <stdarg.h>

#define LOG_PRINTER(x) ((printer_f)x)
#define LOG_STREAM(x) ((void*)x)

/**
 * Poziom na którym ma być logowana notatka
 */
enum LOG_LEV {
	DBG = 0,
	INFO,
	_WARNING,
	_ERROR,
	BUG,
	CRITICAL
};

/**
 * Opisy poziomów logowania
 */
static const char LEV_STRING[][32] =
	{ "[DEBUG]  "
	, "[INFO]   "
	, "[WARNING]"
	, "[ERROR]  "
	, "[BUG]    "
	, "[CRITIC] " };

/**
 * Definicja funkcji która może być użyta do wyświetlania
 * przy pomocy loggera
 */
typedef int (*printer_f)(void* , const char* , va_list );

/**
 * Struktrua przechowująca dane o pojedynczym logu
 */
struct _log_t {
	void**stream;
	printer_f*printer;
	int n;
	enum LOG_LEV lev;
};

typedef struct _log_t log_t;


log_t*log_init();

void log_set_lev( enum LOG_LEV );
void log_fset_lev( log_t* , enum LOG_LEV );

int log_printf( enum LOG_LEV , const char* , ... );
int log_fprintf( log_t* , enum LOG_LEV , const char* , ... );
int log_vfprintf( log_t* , enum LOG_LEV , const char* , va_list args );
void log_add( void*stream , printer_f func );
void log_fadd( log_t*log , void*stream , printer_f func );
void log_del( void*stream );
void log_fdel( log_t*log , void*stream );

#endif /* __LOGGER_H__ */

