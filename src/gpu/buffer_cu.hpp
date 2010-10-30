#include "buffer.h"

namespace GPU
{
	//
	// Cuda buffer declaration
	//
	template<typename T>
	class BufferCu : public BufferBase<T> {
	public:
		BufferCu( );
		BufferCu( const size_t num , const T*data = NULL );
		virtual ~BufferCu();

		virtual void resize( size_t num , const T*data = NULL );

		T* data();
		void bind();
		void unbind();

	protected:
		bool bound;
		T* cuPtr;
	};

	//
	// Cuda buffer definition
	// 
	template<typename T>
	BufferCu<T>::BufferCu( )
		: bound( false )
		, cuPtr( NULL )
	{
	}

	template<typename T>
	BufferCu<T>::BufferCu( const size_t num , const T*data )
		: bound( false )
		, cuPtr( NULL )
	{
		resize( num, data );
	}

	template<typename T>
	BufferCu<T>::~BufferCu()
	{
		//TODO free mem
		assert( !bound );
	}
	
	template<typename T>
	void BufferCu<T>::resize( size_t num , const T*data )
	{
		//TODO implement me
	}

	template<typename T>
	T* BufferCu<T>::data()
	{
		assert( bound );
		return cuPtr;
	}

	template<typename T>
	void BufferCu<T>::bind()
	{
		assert( !bound );
		bound = true;
		// TODO implement me
	}

	template<typename T>
	void BufferCu<T>::unbind()
	{
		assert( bound );
		bound = false;
		// TODO implement me
	}
}
