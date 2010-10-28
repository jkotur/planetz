#ifndef BUFFER_H
#define BUFFER_H

#include <GL/glew.h>

#include <cassert>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

namespace GPU
{

	template<typename T>
	class BufferBase
	{
	public:
		BufferBase() : size(0) , realsize(0) {}
		virtual ~BufferBase() {}
	protected:
		size_t size; // size of pointed data == number of elements, not bytes
		size_t realsize;
		// TODO: void fireEventContentChanged();
	};

	template<typename T>
	class BufferGl : public BufferBase<T> {
	public:
		/** buffer mapped states */
		enum BUFFER_STATE {
			BUF_GL , // opengl
			BUF_CU , // cuda 
			BUF_H    // host 
		};

		BufferGl( );
		BufferGl( const size_t num , const T*data = NULL );
		virtual ~BufferGl( );

		void resize( const size_t num , const T*data = NULL );

		T*     map( enum BUFFER_STATE state );
		void unmap();
		
	protected:
		GLuint glId; // opengl buffer id
		T*     cuPtr;// cuda gpu data pointer
		T*     hPtr; // host mapped opengl gpu data pointer

		enum BUFFER_STATE state;
	};
	
	//
	// Cuda buffer declaration
	//
	template<typename T>
	class BufferCu : public BufferBase<T> {
	public:
		BufferCu ();
		virtual ~BufferCu();

	protected:
		T*     cuPtr;
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
		: glId(0) , cuPtr(NULL) , hPtr(NULL) , state(BUF_GL)
	{
	}

	template<typename T>
	BufferGl<T>::BufferGl( const size_t num , const T*data )
		: glId(0) , cuPtr(NULL) , hPtr(NULL) , state(BUF_GL)
	{
		resize( num , data );
	}

	template<typename T>
	BufferGl<T>::~BufferGl()
	{
		cudaGLUnregisterBufferObject( glId );
		glDeleteBuffers( 1 , &glId );
	}

	template<typename T>
	void BufferGl<T>::resize( const size_t num , const T*data )
	{
		// FIXME: assert if unmapped or unmap?
		assert( state == BUF_GL );

		size_t new_size = num*sizeof(T);

		if( new_size <= this->realsize ) {
			this->size = new_size;
			return;
		}

		if( !glId ) {
			glGenBuffers(1,&glId);
			cudaGLRegisterBufferObject( glId );
		}

		glBindBuffer(GL_ARRAY_BUFFER,glId);
		// FIXME: gl constants are hard-coded, is it good?
		glBufferData(GL_ARRAY_BUFFER,new_size,(GLvoid*)data,GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER,0);

		this->size =this-> realsize = new_size;
	}

	template<typename T>
	T* BufferGl<T>::map( enum BUFFER_STATE new_state )
	{
		if( state == new_state ) return state==BUF_CU?cuPtr:(state==BUF_H?hPtr:NULL);

		unmap();

		state = new_state;

		if( state == BUF_GL ) return NULL;
		if( state == BUF_CU ) { cudaGLMapBufferObject( (void**)&cuPtr , glId ); return cuPtr; }
		if( state == BUF_H  ) { hPtr = (T*)glMapBuffer( GL_ARRAY_BUFFER , glId ); return hPtr; }
		assert(false); // should never reach this code
		return NULL;
	}

	template<typename T>
	void BufferGl<T>::unmap()
	{
		if( state == BUF_GL ) return;
		if( state == BUF_CU ) { cudaGLUnmapBufferObject( glId ); return; }
		if( state == BUF_H  ) { glUnmapBuffer( glId ); return; }
		assert( false ); // should never reach this code
	}
}

#endif // BUFFER_H

