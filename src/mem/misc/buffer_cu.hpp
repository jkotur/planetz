#ifndef _BUFFER_CU_HPP_
#define _BUFFER_CU_HPP_
#include "buffer.h"
#include <debug/routines.h>

namespace MEM
{
namespace MISC
{
	//
	// Cuda buffer declaration
	//
	template<typename T>
	class BufferCu : public BufferBase<T>
	{
	public:
		BufferCu( );
		BufferCu( const size_t num , const T*data = NULL );
		virtual ~BufferCu();

		virtual void resize( size_t num , const T*data = NULL );

		T* h_data();
		T* d_data();
		T getAt(unsigned i);
		void setAt(unsigned i, const T&);

		void bind();
		void unbind();

	protected:
		T* d_cuPtr;
		T* h_cuPtr;

		void device_ptr_free();
		void device_ptr_alloc(unsigned n);
		void device_ptr_assign(const T* data);
	};

	//
	// Cuda buffer definition
	// 
	template<typename T>
	BufferCu<T>::BufferCu( )
		: d_cuPtr( NULL )
		, h_cuPtr( NULL )
	{
	}

	template<typename T>
	BufferCu<T>::BufferCu( const size_t num , const T*data )
		: d_cuPtr( NULL )
		, h_cuPtr( NULL )
	{
		resize( num, data );
	}

	template<typename T>
	BufferCu<T>::~BufferCu()
	{
		log_printf(INFO, ">>> ~BufferCu\n");
		device_ptr_free();
		ASSERT_MSG(!h_cuPtr, "BufferCu bound on destruction!");
		log_printf(INFO, "<<< ~BufferCu\n");
	}
	
	template<typename T>
	void BufferCu<T>::resize( size_t num , const T*data )
	{
		ASSERT( !h_cuPtr );

		device_ptr_free();
		device_ptr_alloc(num);
		device_ptr_assign(data);
	}

	template<typename T>
	T* BufferCu<T>::h_data()
	{
		ASSERT( h_cuPtr );
		return h_cuPtr;
	}

	template<typename T>
	T* BufferCu<T>::d_data()
	{
		ASSERT( d_cuPtr );
		return d_cuPtr;
	}

	template<typename T>
	T BufferCu<T>::getAt(unsigned i)
	{
		T retval;
		T* pRetval = &retval;
		cudaMemcpy(&pRetval, d_cuPtr + i, sizeof(T), cudaMemcpyDeviceToHost);
		DBGPUT( CUT_CHECK_ERROR( "memcpy" ) );
		return retval;
	}

	template<typename T>
	void BufferCu<T>::setAt(unsigned i, const T& val)
	{
		cudaMemcpy(d_cuPtr + i, &val, sizeof(T), cudaMemcpyHostToDevice);
		DBGPUT( CUT_CHECK_ERROR( "memcpy" ) );
	}

	template<typename T>
	void BufferCu<T>::bind()
	{
		return;
		ASSERT( !h_cuPtr );
		h_cuPtr = new T[this->length];
		cudaMemcpy(&h_cuPtr, d_cuPtr, this->length * sizeof(T), cudaMemcpyDeviceToHost );
		DBGPUT( CUT_CHECK_ERROR( "memcpy" ) );
	}

	template<typename T>
	void BufferCu<T>::unbind()
	{
		return;
		ASSERT( h_cuPtr );
		cudaMemcpy(&d_cuPtr, h_cuPtr, this->length * sizeof(T), cudaMemcpyHostToDevice );
		DBGPUT( CUT_CHECK_ERROR( "memcpy" ) );
		delete [] h_cuPtr;
		h_cuPtr = NULL;
	}

	template<typename T>
	void BufferCu<T>::device_ptr_free()
	{
		if( d_cuPtr )
		{
			ASSERT_MSG(false, "  unreachable code reached");
			cudaFree( d_cuPtr );
			DBGPUT( CUT_CHECK_ERROR( "free" ) );
			d_cuPtr = NULL;
		}
	}

	template<typename T>
	void BufferCu<T>::device_ptr_alloc(unsigned num)
	{
		ASSERT( !d_cuPtr );
		return;
		this->length = num;
		this->size = this->realsize = num * sizeof(T);
		cudaMalloc((void**)&d_cuPtr, this->size );
		DBGPUT( CUT_CHECK_ERROR( "malloc" ) );
	}

	template<typename T>
	void BufferCu<T>::device_ptr_assign(const T* data)
	{
		return;
		ASSERT( d_cuPtr );
		if( data )
		{
			cudaMemcpy(&d_cuPtr, data, this->size, cudaMemcpyHostToDevice );
			DBGPUT( CUT_CHECK_ERROR( "memcpy" ) );
		}
	}
}
}
#endif
