#ifndef _BUFFER_CU_HPP_
#define _BUFFER_CU_HPP_
#include "buffer.h"
#include <debug/routines.h>

#define PRINT_OUT_BUF( buf, format ) \
	{ \
		(buf).bind(); \
		for( unsigned i = 0; i < (buf).getLen(); ++i )\
		{log_printf( INFO, #buf"[%u] = "format"\n", i, (buf).h_data()[i]);}\
		(buf).unbind();\
	}

namespace MEM
{
namespace MISC
{
	/**
	 * @brief Bufor danych przechowywanych na karcie graficznej z wykorzystaniem CUDA.
	 */
	template<typename T>
	class BufferCu : public BufferBase<T>
	{
	public:
		BufferCu( );
		/**
		 * @brief Tworzy bufor o podanym rozmiarze.
		 *
		 * @param num Ilość elementów w tworzonym buforze.
		 *
		 * @param data Opcjonalnie - dane do skopiowania.
		 */
		BufferCu( const size_t num , const T*data = NULL );
		virtual ~BufferCu();

		/**
		 * @brief Zmienia rozmiar bufora.
		 *
		 * @param num Nowa ilość elementów.
		 *
		 * @param data Opcjonalnie - dane do skopiowania.
		 */
		virtual void resize( size_t num , const T*data = NULL );

		/**
		 * @brief Wpisuje wartość do bufora. Działa tylko dla buforów jednoelementowych.
		 *
		 * @param val Wartość do wpisania.
		 */
		virtual void assign( T val );

		/**
		 * @brief Pobiera wartość z bufora. Działa tylko dla buforów jednoelementowych.
		 *
		 * @returns Pobrana wartość.
		 */
		virtual T retrieve();

		/**
		 * @brief Dane dostępne z CPU. Wymaga wcześniejszego wywołania metody bind().
		 */
		T* h_data();

		/**
		 * @brief Dane dostępne z GPU.
		 */
		T* d_data();

		/**
		 * @brief Pobiera wartość spod konkretnego indeksu.
		 *
		 * @param i Indeks.
		 *
		 * @returns Wartość skopiowana z GPU.
		 */
		T getAt(unsigned i) const;

		/**
		 * @brief Ustawia wartość pod zadanym indeksem.
		 *
		 * @param i Indeks.
		 *
		 * @param val Wartość do skopiowania na GPU.
		 */
		void setAt( unsigned i, const T& val );

		/**
		 * @brief Udostępnia dane z karty graficznej na CPU. 
		 *
		 * @details Ta metoda musi zostać wywołana przed pobraniem wkaźnika na dane metodą h_data().
		 */
		void bind();

		/**
		 * @brief Zwalnia dane do użycia przez kartę graficzną.
		 *
		 * @details Ta metoda musi zostać wywołana po zakończeniu operacji na danych zwróconych przez h_data(), a przed rozpocząciem korzystania z danych zwracanych przez d_data().
		 */
		void unbind();

	protected:
		/**
		 * @brief Wskaźnik na dane na karcie graficznej.
		 */
		T* d_cuPtr;

		/**
		 * @brief Wskaźnik na dane na CPU.
		 */
		T* h_cuPtr;

		/**
		 * @brief Zwalnia pamięc na GPU.
		 */
		void device_ptr_free();

		/**
		 * @brief Alokuje pamięć na GPU.
		 */
		void device_ptr_alloc(unsigned n);

		/**
		 * @brief Przypisuje dane z CPU do zaalokowanej pamięci na GPU.
		 *
		 * @param data Dane do skopiowania. Wywołanie z NULL nie wykona żadnej akcji.
		 */
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
		device_ptr_free();
		ASSERT_MSG(!h_cuPtr, "BufferCu bound on destruction!");
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
	void BufferCu<T>::assign( T val )
	{
		ASSERT_MSG( 1 == BufferBase<T>::getLen(), "assign() works for one field buffers only!" );
		setAt(0, val);
	}

	template<typename T>
	T BufferCu<T>::retrieve()
	{
		ASSERT_MSG( 1 == BufferBase<T>::getLen(), "retrieve() works for one field buffers only!" );
		return getAt(0);
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
	T BufferCu<T>::getAt(unsigned i) const
	{
		T retval;
		cudaMemcpy(&retval, d_cuPtr + i, sizeof(T), cudaMemcpyDeviceToHost);
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
		ASSERT( !h_cuPtr );
		h_cuPtr = new T[this->length];
		cudaMemcpy(h_cuPtr, d_cuPtr, this->length * sizeof(T), cudaMemcpyDeviceToHost );
		DBGPUT( CUT_CHECK_ERROR( "memcpy" ) );
	}

	template<typename T>
	void BufferCu<T>::unbind()
	{
		ASSERT( h_cuPtr );
		cudaMemcpy(d_cuPtr, h_cuPtr, this->length * sizeof(T), cudaMemcpyHostToDevice );
		DBGPUT( CUT_CHECK_ERROR( "memcpy" ) );
		delete [] h_cuPtr;
		h_cuPtr = NULL;
	}

	template<typename T>
	void BufferCu<T>::device_ptr_free()
	{
		if( d_cuPtr )
		{
			cudaFree( d_cuPtr );
			DBGPUT( CUT_CHECK_ERROR( "free" ) );
			d_cuPtr = NULL;
		}
	}

	template<typename T>
	void BufferCu<T>::device_ptr_alloc(unsigned num)
	{
		ASSERT( !d_cuPtr );
		this->length = num;
		this->size = this->realsize = num * sizeof(T);
		if( 0 == num )
			return;
		cudaMalloc((void**)&d_cuPtr, this->size );
		DBGPUT( CUT_CHECK_ERROR( "malloc" ) );
		ASSERT( d_cuPtr );
	}

	template<typename T>
	void BufferCu<T>::device_ptr_assign(const T* data)
	{
		if( !this->size )
		{
			return;
		}
		ASSERT( d_cuPtr );
		if( data )
		{
			cudaMemcpy(d_cuPtr, data, this->size, cudaMemcpyHostToDevice );
			DBGPUT( CUT_CHECK_ERROR( "memcpy" ) );
		}
	}
}
}
#endif
