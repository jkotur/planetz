#ifndef BUFFER_H
#define BUFFER_H

#include <GL/glew.h>

#include <cassert>

#include <cstdio>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "cuda/err.h"

namespace GPU
{
	typedef unsigned int uint;

	/** buffer mapped states */
	enum BUFFER_STATE {
		BUF_GL , // opengl
		BUF_CU , // cuda 
		BUF_H    // host 
	};

	template<typename T>
	class BufferBase
	{
	public:
		BufferBase() : size(0) , realsize(0) {}
		virtual ~BufferBase() {}

		virtual void resize( size_t num , const T*data = NULL ) =0;

		virtual size_t getSize() const
		{
			return size;
		}

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

	template<typename T>
	class BufferGl : public BufferBase<T> {
		// prevent from copying buffer. If so, smart counter is needed 
		// to unregister from cuda and delete from opengl
		BufferGl( BufferGl<T>& ) {}
	public:
		BufferGl( );
		BufferGl( const size_t num , const T*data = NULL );
		virtual ~BufferGl( );

		virtual void resize( const size_t num , const T*data = NULL );

		T*       map( enum BUFFER_STATE state );
		const T* map( enum BUFFER_STATE state ) const;
		void unmap() const;
		
		void bind() const;
		void unbind() const;

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
		enum BUFFER_STATE*pstate;
	};
	
	//
	// BUF and host data in one buffer (only concept)
	//
	template<typename T,typename BUF>
	class BufferBoth : public BUF
	{
	public:
		BufferBoth( size_t , const T*data = NULL );
		virtual ~BufferBoth();
	protected:
		T*     hostDataPtr;
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
	void BufferGl<T>::resize( const size_t num , const T*data )
	{
		// FIXME: assert if unmapped or unmap?
		assert( state == BUF_GL );

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
		assert( state == BUF_GL );
		return glId;
	}

	template<typename T>
	void BufferGl<T>::bind() const
	{
		assert( state == BUF_GL );
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
		assert( this->size > 0 );

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
		assert(false); // should never reach this code
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
		assert( this->size > 0 );

		
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

#endif // BUFFER_H

