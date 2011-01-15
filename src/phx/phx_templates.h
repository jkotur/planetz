#ifndef _PHX_TEMPLATES_H_
#define _PHX_TEMPLATES_H_
#include "mem/misc/buffer.h"
#include "mem/misc/buffer_cu.hpp"

inline bool operator==(const float3& l, const float3& r)
{
	return 0 == memcmp( &l, &r, sizeof(float3) );
}

template<class T>
class Validator
{
	public:
		void validate( const T& value )
		{
		}
};

template<>
class Validator<float>
{
	public:
		void validate( const float& value )
		{
			//log_printf( DBG, "%f\n", value );
			ASSERT( !isnan(value) );
		}
};

template<>
class Validator<float3>
{
	public:
		void validate( const float3& value )
		{
			//log_printf( DBG, "[%f, %f, %f]\n", value.x, value.y, value.z );
			ASSERT( !isnan(value.x) );
			ASSERT( !isnan(value.y) );
			ASSERT( !isnan(value.z) );
		}
};

template<class T, template<class S> class BUF>
class BufferAdapter
{
	public:
		BufferAdapter( BUF<T>& );

		T* hostData();
};

template<class T>
class BufferAdapter<T, MEM::MISC::BufferGl >
{
	public:
		BufferAdapter( MEM::MISC::BufferGl<T>& b ):buf(b)
		{
		}

		T* hostData()
		{
			return buf.map( MEM::MISC::BUF_H );
		}
	private:
		MEM::MISC::BufferGl<T> &buf;
};

template<class T>
class BufferAdapter<T, MEM::MISC::BufferCu>
{
	public:
		BufferAdapter( MEM::MISC::BufferCu<T>& b ):buf(b)
		{
			buf.bind();
		}
		T *hostData()
		{
			return buf.h_data();
		}
		~BufferAdapter()
		{
			buf.unbind();
		}
	private:
		MEM::MISC::BufferCu<T> &buf;
};

template<class T, template<class S> class BUF>
class ConstChecker
{
	public:
		ConstChecker() : buf(NULL), data(NULL) {}
		virtual ~ConstChecker(){if(data)delete[]data;}
		void setBuf(BUF<T> *b, unsigned _size )
		{
			buf = b;
			size = _size;
			if( data )
			{
				delete []data;
			}
			data = new T[ size ];
			BufferAdapter<T, BUF> ad( *buf );
			ASSERT( ad.hostData() );
			memcpy( data, ad.hostData(), sizeof(T) * size );
			Validator<T> v;
			for( unsigned i = 0; i < size; ++i )
			{
				//log_printf( DBG, "data[%u] ", i );
				v.validate( data[i] );
			}
		}

		void checkBuf()
		{
			T *actual_data = new T[ size ];
			ASSERT( actual_data != NULL );
			BufferAdapter<T, BUF> ad( *buf );
			memcpy( actual_data, ad.hostData(), sizeof(T) * size );
			Validator<T> v;
			for( unsigned i = 0; i < size; ++i )
			{
				//log_printf( DBG, "actual_data[%u] ", i );
				v.validate( actual_data[i] );
				ASSERT( actual_data[i] == data[i] );
			}
			delete []actual_data;
		}
	private:
		BUF<T> *buf;
		unsigned size;
		T *data;
};

#endif // _PHX_TEMPLATES_H_
