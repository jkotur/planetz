#include <stdlib.h>

#include "logger.h"

log_t*LOGGER = log_init();

log_t*log_init()
{
	log_t*out = (log_t*)malloc(sizeof(log_t));
	out->stream = NULL;
	out->printer = NULL;
	out->n = 0;
	out->lev = (enum LOG_LEV)0;
	return out;
}

void log_set_lev( enum LOG_LEV lev )
{
	log_fset_lev(LOGGER,lev);
}

void log_fset_lev( log_t*log , enum LOG_LEV lev )
{
	if( log ) log->lev = lev;
}

void log_add( void*stream , printer_f func )
{
	log_fadd(LOGGER,stream,func);
}

void log_fadd( log_t*log , void*stream , printer_f func )
{
	log->n++;
	log->stream = (void**)realloc(log->stream,sizeof(void*)*log->n);
	log->printer=(printer_f*)realloc(log->printer,sizeof(void*)*log->n);
	log->stream[log->n-1] = stream;
	log->printer[log->n-1]= func;
}

int log_printf( enum LOG_LEV lev , const char*format , ... )
{
	if( !LOGGER ) return -1;
	if( LOGGER->lev > lev ) return 0;
	va_list args;
	va_start( args, format );
	int out = log_vfprintf(LOGGER,lev,format,args);
	va_end(args);
	return out;
}
			
int log_fprintf( log_t*log , enum LOG_LEV lev , const char*format , ... )
{
	if( !log ) return -1;
	if( log->lev > lev ) return 0;
	va_list args;
	va_start( args, format );
	int out = log_vfprintf(log,lev,format,args);
	va_end( args );
	return out;
}

int log_vfprintf( log_t*log , enum LOG_LEV lev , const char*format , va_list args )
{
	int out = 1;
	int i;
	for( i=0 ; i<log->n ; i++ )
	{
		va_list args_copy;
#ifndef _WIN32
		va_copy(args_copy,args);
#else
		args_copy = args;
#endif
		if( log->printer[i]( log->stream[i] , (char*)LEV_STRING[lev] , args_copy ) == -1 
		 || log->printer[i]( log->stream[i] , format , args_copy ) == -1 ) out = -1;
		va_end(args_copy);
	}
	return out;
}

