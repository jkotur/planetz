#include <limits>
#include <cudpp.h>

#include "cuda/err.h"

#include "kmeans.h"
#include "kmeans_kernel.cu"
//#define DEBUG

using std::numeric_limits;
using namespace MEM::MISC;
using namespace PHX;

Clusterer::Clusterer(BufferGl<float3> *positions, BufferCu<float> *masses)
	: m_pPositions( positions )
	, m_pPlanetMasses( masses )
	, m_prevSize( 0 )
{
}

Clusterer::~Clusterer()
{
}

void Clusterer::kmeans()
{
	initClusters();
	float err = numeric_limits<float>::infinity(), prev_err;
	unsigned iters = 0;
	do
	{
		prev_err = err;
		err = compute();
		++iters;
	}
	while( abs(prev_err-err) >  1e-5 * err );
	calcAttributes();
}

namespace
{
void massSelect( BufferCu<float3> *centers, BufferGl<float3> *planets, BufferCu<float> *masses )
{
	float3 *d_planets = planets->map( BUF_CU );
	cudaMemcpy( centers->d_data(), d_planets, centers->getSize(), cudaMemcpyDeviceToDevice );
	planets->unmap();

	TODO("Wpisanie początkowych pozyji klastrów - k najcięższych planet");
}
}

void Clusterer::initClusters()
{
	unsigned n = m_pPositions->getLen();
	if( n == m_prevSize )
	{
		return;
	}
	m_prevSize = n;

	ASSERT( n );
	TODO("Mądre obliczanie k na podstawie n");
	unsigned k = min( 512, 1 + n / 100 );
	m_holder.resize( k, n );
	m_errors.resize( n );
	m_shuffle.resize( n );
	m_counts.resize( k );
	
	massSelect( &m_holder.centers, m_pPositions, m_pPlanetMasses );
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

	CUDPPHandle sortplan;
	CUDPPResult result = cudppPlan(&sortplan, cfg, m_holder.assignments.getLen(), 1, 0);
	ASSERT( result == CUDPP_SUCCESS );
	if (result != CUDPP_SUCCESS)
	{
		log_printf(CRITICAL,"Error creating CUDPPPlan: %d\n", result);
		exit(1);
	}

	/// 9 bitów - magiczna stała wynika z faktu, że w tej chwili może i tak istnieć najwyżej 512 klastrów, co oznacza, że numer klastra mieści się w 9 bitach.
	ASSERT( m_holder.assignments.getLen() == m_shuffle.getLen() );
	ASSERT( m_holder.assignments.getSize() == m_shuffle.getSize() );
	ASSERT( m_shuffle.getSize() == sizeof(unsigned int) * m_shuffle.getLen() );
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
		reduceSelective_f3<512> <<< grid, block, mem >>>(m_pPositions->map( BUF_CU ), m_holder.centers.d_data(), m_counts.d_data(), i, m_shuffle.d_data());
		CUT_CHECK_ERROR( "Kernel launch" );
	}
}

void Clusterer::calcAttributes()
{
	// DELICIOUS COPY PASTA // TODO: zrobić to kiedyś ładnie
	for(unsigned i = 0; i < m_holder.k_size(); ++i)
	{
		dim3 block( 512 );
		dim3 grid( 1 );
		unsigned mem = sizeof(float) * 512;
		sumSelective_f<512> <<< grid, block, mem >>>(m_pPlanetMasses->d_data(), m_holder.masses.d_data(), m_counts.d_data(), i, m_shuffle.d_data());
		CUT_CHECK_ERROR( "Kernel launch" );
	}
}

MEM::MISC::BufferCu<unsigned>* Clusterer::getCounts()
{
	return &m_counts;
}

MEM::MISC::BufferCu<unsigned>* Clusterer::getShuffle()
{
	return &m_shuffle;
}

MEM::MISC::BufferCu<float3>* Clusterer::getCenters()
{
	return &m_holder.centers;
}

MEM::MISC::BufferCu<float>* Clusterer::getMasses()
{
	return &m_holder.masses;
}
