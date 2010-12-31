#ifndef BUFFER_H
#define BUFFER_H

#include <GL/glew.h>

#include <cassert>

#include <cstdio>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "cuda/err.h"

#include "debug/routines.h"

namespace MEM
{
namespace MISC
{
	typedef unsigned int uint;

	/** możliwe stany bufforów */
	enum BUFFER_STATE {
		BUF_GL , /**< opengl */
		BUF_CU , /**< cuda   */
		BUF_H    /**< host   */
	};

	/** 
	 * @brief Bazowa klasa dla buforów pamięci
	 */
	template<typename T>
	class BufferBase
	{
	public:
		/** 
		 * @brief Konstruktor pustego bufora
		 */
		BufferBase() : length(0), size(0) , realsize(0){}
		virtual ~BufferBase() {}

		/** 
		 * @brief Funkcja zmieniająca wielkość bufora.
		 * 
		 * @param num nowa wielkość bufora w byte'ach.
		 * @param data dodatkowo można podać dane które mają być skopiowane do bufora
		 */
		virtual void resize( size_t num , const T*data = NULL ) =0;
		/** 
		 * @brief Funkcja przypisująca wartość dla jednoelementowych buforów.
		 * 
		 * @param val wartość do wpisania
		 */
		virtual void assign( T val ) = 0; // for one-element buffers - set its value to val

		/** 
		 * @brief Zwraca wielkość bufora.
		 * 
		 * @return wielkość bufora w bajtach
		 */
		virtual size_t getSize() const
		{
			return size;
		}

		/** 
		 * @brief Zwraca długość bufora.
		 * 
		 * @return wielkość bufora w ilości elementów
		 */
		virtual uint   getLen() const
		{
			return length;
		}
	protected:
		uint   length; // number of Ts in buffer
		size_t size; // size of pointed data == number of bytes
		size_t realsize;
		// TODO: void fireEventContentChanged();
	};

	/** 
	 * @brief Buffor trzymający pamięć na karcie graficznej przy pomocy OpenGLa.
	 * API OpenGLa pozwala na mapowanie takich bufforów do pamięci hosta,
	 * czyli w tym przypadku CPU, natomiast API CUDA pozwala na mapowanie ich
	 * do pamięci CUDA. Są one więc najbardziej wszechstronnymi buforami karty graficznej.
	 */
	template<typename T>
	class BufferGl : public BufferBase<T> {
		// prevent from copying buffer. If so, smart counter is needed 
		// to unregister from cuda and delete from opengl
		BufferGl( const BufferGl<T>& );
	public:
		BufferGl( );
		BufferGl( const size_t num , const T*data = NULL );
		virtual ~BufferGl( );

		virtual void resize( const size_t num , const T*data = NULL );
		virtual void assign( T val );
		virtual T retrieve();

		/** 
		 * @brief mapuje bufor. Możliwe mapowania to OpenGL, CUDA lub HOST
		 * 
		 * @param state nowy stan bufora
		 * 
		 * @return wskaźnik do pamięci na którą zmapowany został bufor.
		 */
		T*       map( enum BUFFER_STATE state );
		/** 
		 * @brief stała wersja funkcji mapującaej. Zwraca stały wskaźnik.
		 * 
		 * @param state nowy stan bufora.
		 * 
		 * @return stały wskaźnik do pamięci.
		 */
		const T* map( enum BUFFER_STATE state ) const;
		/** 
		 * @brief przywraca buffor do domyślego stanu. Po wywołaniu tej funckji
		 * wszystkie wskaźniki otrzymane przez wywołanie map stają się nieaktualne.
		 */
		void unmap() const;
		
		/** 
		 * @brief binduje buffor do użycia przez opengl jako GL_ARRAY_BUFFER
		 */
		void bind() const;
		/** 
		 * @brief ustawia pusty bufor jako GL_ARRAY_BUFFER
		 */
		void unbind() const;

		/** 
		 * @brief Zwraca id bufora w API OpenGLa
		 * 
		 * @return id bufora.
		 */
		GLuint getId() const;
	protected:
		T* fucking_no_const_cast_workaround( enum BUFFER_STATE state ) const;

		void gl_resize( const size_t new_size , const T*data );

		GLuint glId; // opengl buffer id
		T*     cuPtr;// cuda gpu data pointer
		T*     hPtr; // host mapped opengl gpu data pointer

		enum BUFFER_STATE  state;

		// nasty const hacks
		T**   phPtr;
		enum BUFFER_STATE* const pstate;
	};
	
	//
	// Implementation BufferGl
	//
	template<typename T>
	BufferGl<T>::BufferGl( )
		: glId(0) , cuPtr(NULL) , hPtr(NULL) , state(BUF_GL) ,
		  phPtr(&hPtr) , pstate(&state)
	{
	}

	template<typename T>
	BufferGl<T>::BufferGl( const size_t num , const T*data )
		: glId(0) , cuPtr(NULL) , hPtr(NULL) , state(BUF_GL) ,
		  phPtr(&hPtr) , pstate(&state)
	{
		resize( num , data );
	}

	template<typename T>
	BufferGl<T>::~BufferGl()
	{
		if( glId ) {
			log_printf(DBG,"[DEL] Not empty GL buffer\n");
			unmap();
			cudaGLUnregisterBufferObject( glId );
			CUT_CHECK_ERROR("Unregistering buffer while deleting BufferGl");
			glDeleteBuffers( 1 , &glId );
		}
	}

	template<typename T>
	void BufferGl<T>::gl_resize( const size_t new_size , const T*data )
	{
		log_printf(DBG,"Resizing gl buffer from %d bytes to %d bytes\n",this->realsize,new_size);
		glBindBuffer(GL_ARRAY_BUFFER,glId);
		// FIXME: gl constants are hard-coded, is it good?
		glBufferData(GL_ARRAY_BUFFER,new_size,(GLvoid*)data,GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER,0);
	}

	template<typename T>
	void BufferGl<T>::assign( T val )
	{
		ASSERT( 1 == BufferBase<T>::getLen() );
		map( BUF_H )[0] = val;
		unmap();
	}

	template<typename T>
	T BufferGl<T>::retrieve()
	{
		ASSERT( 1 == BufferBase<T>::getLen() );
		T retval = map( BUF_H )[0];
		unmap();
		return retval;
	}

	template<typename T>
	void BufferGl<T>::resize( const size_t num , const T*data )
	{
		// FIXME: assert if unmapped or unmap?
		ASSERT_MSG( state == BUF_GL , "Buffer not in gl mode\n" );

		size_t new_size = num*sizeof(T);

		if( new_size <= this->realsize ) {
			this->length = num;
			this->size = new_size;
			return;
		}

		if( !glId ) {
			glGenBuffers(1,&glId);

			gl_resize( new_size , data );

			cudaGLRegisterBufferObject( glId );
			CUT_CHECK_ERROR("Registering buffer while resizing BufferGl");
		} else
			gl_resize( new_size , data );

		this->length = num;
		this->size =this-> realsize = new_size;
	}

	template<typename T>
	GLuint BufferGl<T>::getId() const
	{
		ASSERT( state == BUF_GL );
		return glId;
	}

	template<typename T>
	void BufferGl<T>::bind() const
	{
		ASSERT( state == BUF_GL );
		glBindBuffer( GL_ARRAY_BUFFER , glId );
		// FIXME: another state?
//                state = BUF_BIND;
	}

	template<typename T>
	void BufferGl<T>::unbind() const
	{
		glBindBuffer( GL_ARRAY_BUFFER , 0 );
	}

	template<typename T>
	T* BufferGl<T>::fucking_no_const_cast_workaround( enum BUFFER_STATE new_state ) const
	{
		ASSERT_MSG( this->size > 0 , "Cannot map empty buffer:<" );

		if( state == new_state ) return *pstate==BUF_CU?cuPtr:(state==BUF_H?hPtr:NULL);

		unmap();

		*pstate = new_state;

		if( state == BUF_GL ) return NULL;
		if( state == BUF_CU ) {
			cudaGLMapBufferObject( (void**)&cuPtr , glId );
			CUT_CHECK_ERROR("Mapping BufferGl to cuda\n");
			return cuPtr;
		}
		if( state == BUF_H  ) {
			glBindBuffer( GL_ARRAY_BUFFER , glId );
			*phPtr = (T*)glMapBuffer( GL_ARRAY_BUFFER , GL_READ_WRITE );
			glBindBuffer( GL_ARRAY_BUFFER , 0 );
			return hPtr;
		}
		ASSERT(false); // should never reach this code
		return NULL;
	}

	template<typename T>
	const T* BufferGl<T>::map( enum BUFFER_STATE new_state ) const
	{
		return fucking_no_const_cast_workaround( new_state );
	}

	template<typename T> 
	T* BufferGl<T>::map( enum BUFFER_STATE new_state ) 
	{
		/* fucking const_cast seg faults while casting invalid pointers.
		 * unfortunetely all gpu pointers are invalid for cpu!!
		 */
//                return const_cast<T*>(map(new_state)); 
		return fucking_no_const_cast_workaround( new_state );
	}

	template<typename T>
	void BufferGl<T>::unmap() const
	{
		ASSERT( this->size > 0 );

		
		if( state == BUF_CU ) {
			cudaGLUnmapBufferObject( glId );
			CUT_CHECK_ERROR("Unmapping cuda buffer to BufferGl");
		}
		if( state == BUF_H  ) {
			glBindBuffer( GL_ARRAY_BUFFER , glId );
			glUnmapBuffer( GL_ARRAY_BUFFER );
			glBindBuffer( GL_ARRAY_BUFFER , 0 );
		}

		*pstate = BUF_GL;
	}

}
}
#endif // BUFFER_H

