#include <limits>
#include <cudpp.h>

#include "cuda/err.h"

#include "kmeans.h"
#include "kmeans_kernel.cu"
//#define DEBUG

using std::numeric_limits;
using namespace MEM::MISC;
using namespace PHX;

Clusterer::Clusterer(PhxPlanetFactory *ppf)
	: m_planets( ppf )
	, m_prevSize( 0 )
{
}

Clusterer::~Clusterer()
{
}

void Clusterer::kmeans()
{
	initClusters();
	initCudpp();
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
	termCudpp();
}

/// @todo posprzątać tutaj
void Clusterer::massSelect()
{
	unsigned n = m_planets->size();
	BufferCu<float> masses_copy( n );
	BufferCu<unsigned> indices( n );
	cudaMemcpy( masses_copy.d_data(), m_planets->getMasses().d_data(), sizeof(float) * n, cudaMemcpyDeviceToDevice );
	dim3 block( min( n, 512 ) );
	dim3 grid( 1 + ( n - 1 ) / block.x );
	kmeans__prepare_kernel<<<grid, block>>>( indices.d_data(), n );
	CUT_CHECK_ERROR("kernel launch");
	cudppSort( sortplan, masses_copy.d_data(), indices.d_data(), 8 * sizeof(float), n );
	
	float3 *d_planets = m_planets->getPositions().map( BUF_CU );
	unsigned k = m_holder.centers.getLen();
	block.x = min( 512, k );
	grid.x = 1 + ( k - 1 )/block.x;
	assignCenters<<<grid, block>>>( m_holder.centers.d_data(), d_planets, indices.d_data() + (n - k), k );
	CUT_CHECK_ERROR("kernel launch");
	m_planets->getPositions().unmap();
}

void Clusterer::initClusters()
{
	unsigned n = m_planets->size();
	if( n == m_prevSize )
	{
		return;
	}
	m_prevSize = n;

	ASSERT( n );
	TODO("Mądre obliczanie k na podstawie n");
	unsigned k = min( 512, 1 + n / 1000 );
	m_holder.resize( k, n );
	m_errors.resize( n );
	m_shuffle.resize( n );
	m_counts.resize( k );

	initCudpp( false );
	massSelect();
	termCudpp();
}

size_t Clusterer::getCount() const 
{
	return m_holder.k_size();
}

float Clusterer::compute()
{
	unsigned threads = m_planets->size();
	unsigned k = getCount();
	dim3 block( min( threads, 512 ) );
	dim3 grid( 1 + ( threads - 1 ) / block.x );
	unsigned mem = sizeof(float3) * block.x;
	
	kmeans__findbest_kernel<<< grid, block, mem >>>( k,
		m_holder.centers.d_data(),
		m_planets->getPositions().map( BUF_CU ),
		m_planets->size(),
		m_holder.assignments.d_data(),
		m_errors.d_data() );
	CUT_CHECK_ERROR( "Kernel launch - find best" );
	
	kmeans__prepare_kernel<<< grid, block >>>( m_shuffle.d_data(), m_planets->size() );
	CUT_CHECK_ERROR( "Kernel launch - prepare" );

	sortByCluster();

	kmeans__count_kernel<<< grid, block >>>( m_holder.assignments.d_data() , m_counts.d_data(), m_planets->size() - 1, k );
	CUT_CHECK_ERROR( "Kernel launch - count" );

	reduceMeans();
	return reduceErrors();
}

void Clusterer::sortByCluster()
{
	unsigned n = m_holder.assignments.getLen();
	unsigned bitCount = 2;
	while( (n-1) >> bitCount ) ++bitCount;
	ASSERT( 1u << bitCount >= n );
	cudppSort( sortplan, m_holder.assignments.d_data(), m_shuffle.d_data(), bitCount, n );
}

float Clusterer::reduceErrors()
{
	dim3 grid( 1 );
	dim3 block( 512 );
	//TODO( "tablica out != in" ) !!!!!
	reduceFull<float, 512><<< grid, block>>>(m_errors.d_data(), m_errors.d_data(), m_errors.getLen());
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
		avgSelective_f3<512> <<< grid, block, mem >>>(m_planets->getPositions().map( BUF_CU ), m_holder.centers.d_data(), m_counts.d_data(), i, m_shuffle.d_data());
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
		sumSelective_f<512> <<< grid, block, mem >>>(m_planets->getMasses().d_data(), m_holder.masses.d_data(), m_counts.d_data(), i, m_shuffle.d_data());
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

void Clusterer::initCudpp( bool uint )
{
	CUDPPConfiguration cfg;
	cfg.datatype = uint ? CUDPP_UINT : CUDPP_FLOAT;
	cfg.algorithm = CUDPP_SORT_RADIX;
	cfg.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
	cfg.op = CUDPP_MIN;

	CUDPPResult result = cudppPlan(&sortplan, cfg, m_planets->size(), 1, 0);
	ASSERT( result == CUDPP_SUCCESS );
	if (result != CUDPP_SUCCESS)
	{
		log_printf(CRITICAL,"Error creating CUDPPPlan: %d\n", result);
		exit(1);
	}
}

void Clusterer::termCudpp()
{
	cudppDestroyPlan( sortplan );
}
