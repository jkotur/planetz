#include "kmeans.h"
#include "kmeans_kernel.cu"

#include <cmath>
#include <cstdio>
#include <cassert>
#include <limits>

#include <GL/glew.h>
#include <GL/gl.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <cudpp.h>

#include "cuda/err.h"
//#define DEBUG

using std::numeric_limits;
/* // temporary comment out 
PointSet::PointSet()
{
	means = new BufferCu<float3>();
	counts = new BufferCu<unsigned>();
	assignments = new BufferCu<unsigned>();
	shuffle = new BufferCu<unsigned>();
	errors = new BufferCu<float>();
	buf = new BufferGl();
}

PointSet::~PointSet()
{
	delete means;
	delete counts;
	delete assignments;
	delete shuffle;
	delete errors;
	delete buf;
}

void PointSet::randomize()
{
	float3*d_t;
	cudaGLMapBufferObject( (void**)&d_t , buf->vbo );
	CUT_CHECK_ERROR( "map buffer" );

	cudaMemset( d_t , 0 , sizeof(float3)*buf->len );

	randomize( d_t , buf->len );

	cudaGLUnmapBufferObject( buf->vbo );
	CUT_CHECK_ERROR( "unmap buffer" );

	cudaGLMapBufferObject( (void**)&d_t , buf->cbo );
	CUT_CHECK_ERROR( "map color buffer" );

	cudaMemset( d_t, 0 , sizeof(float)*buf->real_len*3);
	dim3 block( 512 );
	dim3 grid( ceil( buf->len / 512.0f ) );

	kmeans__paint_kernel<<< grid, block >>>(d_t, buf->len, 0.8f, 0.95f, 0.7f);
	CUT_CHECK_ERROR( "Kernel launch" );

	cudaGLUnmapBufferObject( buf->cbo );
	CUT_CHECK_ERROR( "unmap color buffer" );

	assignments->resize( buf->len );
	shuffle->resize( buf->len );
	errors->resize( buf->len );
}

void PointSet::randomize(float3* d_t, unsigned size)
{
	static unsigned *d_seed = NULL;
	if( !d_seed )
	{
		unsigned h_seed = 31337;
		cudaMalloc( (void**)&d_seed , sizeof(unsigned) );
		cudaMemcpy( d_seed, &h_seed , sizeof(unsigned) , cudaMemcpyHostToDevice );
	}
	unsigned threads = size / 100;
	dim3 block( min( 511 , threads ) + 1 );
	dim3 grid( threads / block.x + 1 );

	kmeans__randomize_kernel<<< grid, block >>>(d_t, size, d_seed, width, height, height);
	CUT_CHECK_ERROR( "Kernel launch");
}

void PointSet::kmeans( unsigned k )
{
	float3 *d_points, *d_colors;
#ifdef DEBUG
	unsigned* h_assignments = new unsigned[buf->len];
	unsigned i = 0;
#endif
	means->resize( k );
	counts->resize( k );

	cudaGLMapBufferObject( (void**)&d_points , buf->vbo );
	CUT_CHECK_ERROR( "map buffer" );

	randomize( means->d_ptr, k );

	dim3 block( 512 );
	dim3 grid( ceil( buf->len / 512.0f ) );
	unsigned mem = sizeof(float3) * k;
	float h_err = numeric_limits<float>::infinity(), h_prev_err;

	do
	{
		kmeans__findbest_kernel<<< grid, block, mem >>>( k, means->d_ptr, d_points, buf->len, assignments->d_ptr, errors->d_ptr );
		CUT_CHECK_ERROR( "Kernel launch" );
		kmeans__prepare_kernel<<< grid, block >>>(shuffle->d_ptr, buf->len);
		sortByCluster();
#ifdef DEBUG
		//log_printf(DBG, "Before count_kernel:\n");
		cudaMemcpy( h_assignments, assignments->d_ptr, sizeof(unsigned) * buf->len, cudaMemcpyDeviceToHost );
		for(unsigned j = 0 ; j < buf->len ; ++j )
		{
		//	log_printf(DBG, "{%d} assignments[%d] = %d\n", i, j, h_assignments[j]);
			if( j && h_assignments[j - 1 ] > h_assignments[j] )
			{
				log_printf(CRITICAL, "Nieposortowane dane: [%d]%d > [%d]%d\n", j-1, h_assignments[j - 1] , j, h_assignments[j]);
				for(unsigned k = 0 ; k < buf->len ; ++k )
					log_printf(DBG, "{%d} assignments[%d] = %d\n", i, k, h_assignments[k]);
				exit(-1);
			}
		}
		++i;
#endif
		kmeans__count_kernel<<< grid, block >>>( assignments->d_ptr , counts->d_ptr, buf->len - 1, k );
		CUT_CHECK_ERROR( "Kernel launch" );
		reduceMeans(d_points, k);
		h_prev_err = h_err;
		h_err = reduceErrors();
#ifdef DEBUG
		log_printf(DBG, "Błąd: %f, poprzedni: %f\n", h_err, h_prev_err);
#endif
	}
	while( abs(h_prev_err - h_err) > EPSILON * h_err );

	cudaGLMapBufferObject( (void**)&d_colors , buf->cbo );
	CUT_CHECK_ERROR( "map color buffer" );
	
	kmeans__paint2_kernel<<< grid, block >>>( assignments->d_ptr , d_colors , buf->len , shuffle->d_ptr );
	CUT_CHECK_ERROR( "Kernel launch" );
	
	cudaGLUnmapBufferObject( buf->cbo );
	CUT_CHECK_ERROR( "unmap color buffer" );

	cudaGLUnmapBufferObject( buf->vbo );
	CUT_CHECK_ERROR( "unmap buffer" );
#ifdef DEBUG
	delete h_assignments;
#endif
}

float PointSet::reduceErrors()
{
	dim3 grid( 1 );
	dim3 block( 512 );
	float err;
	unsigned mem = sizeof(float) * 512;
	reduceFull<float, 512><<< grid, block, mem>>>(errors->d_ptr, errors->d_ptr, errors->len);
	CUT_CHECK_ERROR( "Kernel launch" );
	cudaMemcpy( &err, errors->d_ptr, sizeof(float), cudaMemcpyDeviceToHost );
	return err;
}
*/
void PointSet::sortByCluster()
{
	CUDPPConfiguration cfg;
/*	cfg.datatype = CUDPP_UINT;
	cfg.algorithm = CUDPP_SORT_RADIX;
	cfg.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
	cfg.op = CUDPP_MIN;

	CUDPPHandle sortplan = 0;
	CUDPPResult result = cudppPlan(0, &sortplan, cfg, buf->len, 1, 0);
	if (result != CUDPP_SUCCESS)
	{
		log_printf(CRITICAL,"Error creating CUDPPPlan: %d\n", result);
		exit(1);
	}

	cudppSort( sortplan, assignments->d_ptr, shuffle->d_ptr, 8 * sizeof(unsigned), buf->len );
	
	cudppDestroyPlan(sortplan);*/
}
/*
void PointSet::reduceMeans(float3* d_points, unsigned k)
{
	for(unsigned i = 0; i < k; ++i)
	{
		dim3 block( 512 );
		dim3 grid( 1 );
		unsigned mem = sizeof(float3) * 512;
		reduceSelective<512> <<< grid, block, mem >>>(d_points, means->d_ptr, counts->d_ptr, i, shuffle->d_ptr);
		CUT_CHECK_ERROR( "Kernel launch" );
	}
}*/
