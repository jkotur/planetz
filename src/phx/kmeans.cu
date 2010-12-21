#include "kmeans.h"
#include "kmeans_kernel.cu"

#include <cmath>
#include <cstdio>
#include <limits>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <cudpp.h>

#include "cuda/err.h"
//#define DEBUG

using std::numeric_limits;
using namespace MEM::MISC;
using namespace PHX;

Clusterer::Clusterer(BufferGl<float3> *positions)
	: m_pPositions( positions )
{
}

Clusterer::~Clusterer()
{
}

void Clusterer::kmeans()
{
	initClusters();
	float err = numeric_limits<float>::infinity(), prev_err;
	do
	{
		prev_err = err;
		err = compute();
	}
	while( abs(prev_err-err) > EPSILON * err );
}

void Clusterer::initClusters()
{
	TODO("Mądre obliczanie k na podstawie n");
	unsigned n = m_pPositions->getLen();
	unsigned k = min( 512, n / 1000 );
	m_holder.resize( k, n );
	m_errors.resize( n );
	m_shuffle.resize( n );
	m_counts.resize( k );
	m_holder.centers.bind();
	for( size_t i = 0; i < k; ++i )
	{
		m_holder.centers.h_data()[i] = make_float3( 100 * i, 200 * i, i );
		float3 f = m_holder.centers.h_data()[i];
		log_printf( DBG, "centers[%u] = %f, %f, %f\n", i,f.x,f.y,f.z );
	}
	m_holder.centers.unbind();
	TODO("Wpisanie początkowych pozyji klastrów");
}

size_t Clusterer::getCount() const 
{
	return m_holder.k_size();
}

float Clusterer::compute()
{
	unsigned threads = m_pPositions->getLen();
	unsigned k = getCount();
	dim3 block( min( threads, 512 ) );
	dim3 grid( 1 + ( threads - 1 ) / block.x );
	unsigned mem = sizeof(float3) * block.x;
	
	kmeans__findbest_kernel<<< grid, block, mem >>>( k,
		m_holder.centers.d_data(),
		m_pPositions->map( BUF_CU ),
		m_pPositions->getLen(),
		m_holder.assignments.d_data(),
		m_errors.d_data() );
	CUT_CHECK_ERROR( "Kernel launch - find best" );

	kmeans__prepare_kernel<<< grid, block >>>( m_shuffle.d_data(), m_pPositions->getLen() );
	CUT_CHECK_ERROR( "Kernel launch - prepare" );

	sortByCluster();

	kmeans__count_kernel<<< grid, block >>>( m_holder.assignments.d_data() , m_counts.d_data(), m_pPositions->getLen() - 1, k );
	CUT_CHECK_ERROR( "Kernel launch - count" );

	reduceMeans();
	return reduceErrors();
}

void Clusterer::sortByCluster()
{
	CUDPPConfiguration cfg;
	cfg.datatype = CUDPP_UINT;
	cfg.algorithm = CUDPP_SORT_RADIX;
	cfg.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
	cfg.op = CUDPP_MIN;

	CUDPPHandle sortplan = 0;
	CUDPPResult result = cudppPlan(&sortplan, cfg, m_holder.assignments.getLen(), 1, 0);
	ASSERT( result == CUDPP_SUCCESS );
	if (result != CUDPP_SUCCESS)
	{
		log_printf(CRITICAL,"Error creating CUDPPPlan: %d\n", result);
		exit(1);
	}

	/// 9 bitów - magiczna stała wynika z faktu, że w tej chwili może i tak istnieć najwyżej 512 klastrów, co oznacza, że numer klastra mieści się w 9 bitach.
	cudppSort( sortplan, m_holder.assignments.d_data(), m_shuffle.d_data(), 9, m_holder.assignments.getLen() );
	
	cudppDestroyPlan(sortplan);
}

float Clusterer::reduceErrors()
{
	dim3 grid( 1 );
	dim3 block( 512 );
	unsigned mem = sizeof(float) * 512;
	reduceFull<float, 512><<< grid, block, mem>>>(m_errors.d_data(), m_errors.d_data(), m_errors.getLen());
	CUT_CHECK_ERROR( "Kernel launch" );
	return m_errors.getAt(0);
}

void Clusterer::reduceMeans()
{
	for(unsigned i = 0; i < m_holder.k_size(); ++i)
	{
		dim3 block( 512 );
		dim3 grid( 1 );
		unsigned mem = sizeof(float3) * 512;
		reduceSelective<512> <<< grid, block, mem >>>(m_pPositions->map( BUF_CU ), m_holder.centers.d_data(), m_counts.d_data(), i, m_shuffle.d_data());
		CUT_CHECK_ERROR( "Kernel launch" );
	}
}
