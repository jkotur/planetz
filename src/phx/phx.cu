#include <limits>
#include "phx.h"
#include "phx_kernels.h"
#include "phx_templates.h"
#include "kmeans.h"
#include "cuda/math.h"

#define MIN_THREADS 384

using namespace PHX;

ConstChecker<float3, MEM::MISC::BufferGl> pos_checker;
ConstChecker<float, MEM::MISC::BufferCu> mass_checker;
ConstChecker<float3, MEM::MISC::BufferCu> vel_checker;

/**
 * @brief Implementacja klasy Phx.
 *
 * @details Ukrywa szczegóły implementacji, aby nie były widoczne w publicznym interfejsie.
 */
class Phx::CImpl
{
	public:
		/**
		 * @brief Inicjalizacja fizyki
		 */
		CImpl(MEM::MISC::PhxPlanetFactory *p);
		virtual ~CImpl();

		/**
		 * @brief Implementacja Phx::compute
		 */
		void compute(unsigned n);

		/**
		 * @brief Implementacja Phx::enableClusters
		 */
		void enableClusters(bool orly);

		/**
		 * @brief Implementacja Phx::clustersEnabled
		 */
		bool clustersEnabled() const;

		void registerCleaner( MEM::MISC::PlanetHolderCleaner *c );

	private:
		void map_buffers();
		void unmap_buffers();

		void run_nbodies( unsigned planet_count );
		void run_nbodies2();
		void run_nbodies_for_clusters();
		void run_clusters();

		void update_positions();
		void handle_collisions();

		MEM::MISC::PhxPlanetFactory *planets;
		Clusterer clusterer;

		MEM::MISC::BufferCu<unsigned> merges1;
		MEM::MISC::BufferCu<unsigned> merges2;
		MEM::MISC::BufferCu<float3> tmp_vel;
		bool clusters_on;

		MEM::MISC::PlanetHolderCleaner *cleaner;
};

Phx::CImpl::CImpl(MEM::MISC::PhxPlanetFactory *p)
	: planets( p )
	, clusterer( p )
	, clusters_on( true )
	, cleaner( NULL )
{
}

Phx::CImpl::~CImpl()
{
}

void Phx::CImpl::compute(unsigned n)
{
	unsigned planet_count;
	if( !(planet_count = planets->size()) )
		return;
	map_buffers();
	for(unsigned i = 0; i < n; ++i)
	{
		vel_checker.setBuf( &planets->getVelocities(), planet_count );
		mass_checker.setBuf( &planets->getMasses(), planet_count );
		pos_checker.setBuf( &planets->getPositions(), planet_count );
		run_clusters();
		pos_checker.checkBuf();
		mass_checker.checkBuf();
		vel_checker.checkBuf();
		run_nbodies( planet_count );
		update_positions();
	}
	handle_collisions();
	unmap_buffers();
}

void Phx::CImpl::map_buffers()
{
	planets->getPositions().map( MEM::MISC::BUF_CU );
	planets->getRadiuses().map( MEM::MISC::BUF_CU );
	planets->getCount().map( MEM::MISC::BUF_CU );
	merges1.resize( planets->size() );
	merges2.resize( planets->size() );
	tmp_vel.resize( planets->size() ); // prawdopodobnie wystarczy mniej, bo teraz to już tylko dla klastrów jest
}

void Phx::CImpl::unmap_buffers()
{
	planets->getPositions().unmap();
	planets->getRadiuses().unmap();
	planets->getCount().unmap();
}
void Phx::CImpl::run_nbodies2()
{
	unsigned clusters = clusterer.getCount();
	unsigned *h_counts = new unsigned[ clusters ];
	clusterer.getCounts()->bind();
	memcpy( h_counts, clusterer.getCounts()->h_data(), clusters * sizeof(unsigned) );
	clusterer.getCounts()->unbind();
#ifdef PHX_DEBUG
	MEM::MISC::BufferCu<unsigned> whois( planets->size() );
	cudaMemset( whois.d_data(), 0, planets->size() * sizeof(unsigned) );
#endif

	for( unsigned c = 0, prev_count = 0; c < clusters; ++c ) // TODO: odpalić te kernele jednocześnie?
	{
		unsigned threads = h_counts[c] - prev_count;
		if( threads == 0 )
			continue;
#ifdef PHX_DEBUG
	float3 *d_dvs;
	cudaMalloc( &d_dvs, threads * sizeof(float3) );
#endif
		dim3 block( min( threads, MIN_THREADS ) );
		dim3 grid( 1 + ( threads - 1 ) / block.x );
		inside_cluster_interaction<<<grid, block>>>(
			planets->getPositions().map(MEM::MISC::BUF_CU),
			planets->getMasses().d_data(),
			planets->getVelocities().d_data(),
			clusterer.getShuffle()->d_data(),
			clusterer.getCounts()->d_data(),
			c // cluster id
#ifdef PHX_DEBUG
			, d_dvs, 1,whois.d_data()
#endif
			);
		CUT_CHECK_ERROR("Kernel launch");
		prev_count = h_counts[c];
#ifdef PHX_DEBUG
	float3 *dvs = new float3[ threads ];
	cudaMemcpy( dvs, d_dvs, threads * sizeof(float3), cudaMemcpyDeviceToHost );
	float3 sum_dvs = make_float3(0,0,0);
	for( unsigned i = 0; i < threads; ++i ) sum_dvs += dvs[i];
	std::string err = getErr();
	if( !err.empty() )
	{
		log_printf( _ERROR, "CUDA assertion failed: '%s'\n", err.c_str() );
		NOENTRY();
	}
	delete[] dvs;
#endif
	}
	delete[] h_counts;
}

void Phx::CImpl::run_nbodies( unsigned threads )
{	
	ASSERT( threads );
	if( clusters_on )
	{
		run_nbodies2();
		run_nbodies_for_clusters();
		return; // taaa, brzydkie, kiedyś będzie ładniej
	}
	dim3 block( min( threads, MIN_THREADS ) );
	dim3 grid( 1 + (threads - 1) / block.x );
	//unsigned mem = block.x * ( sizeof(float3) + sizeof(float) );

#ifdef PHX_DEBUG
	float3 *d_dvs;
	cudaMalloc( &d_dvs, threads * sizeof(float3) );
#endif
	basic_interaction<<<grid, block>>>( 
		planets->getPositions().map(MEM::MISC::BUF_CU), 
		planets->getMasses().d_data(), 
		planets->getVelocities().d_data(),
		planets->getCount().map(MEM::MISC::BUF_CU)
#ifdef PHX_DEBUG
		, d_dvs, 4210
#endif
		);
	
	CUT_CHECK_ERROR("Kernel launch");
	
#ifdef PHX_DEBUG
	float3 *dvs = new float3[ threads ];
	cudaMemcpy( dvs, d_dvs, threads * sizeof(float3), cudaMemcpyDeviceToHost );
	std::string err = getErr();
	if( !err.empty() )
	{
		log_printf( _ERROR, "CUDA assertion failed: '%s'\n", err.c_str() );
		NOENTRY();
	}
	delete[] dvs;
#endif
}

void Phx::CImpl::run_nbodies_for_clusters()
{
	unsigned threads = clusterer.getCount();
	ASSERT( threads <= MIN_THREADS );
	dim3 block( min( threads, MIN_THREADS ) );
	dim3 grid( 1 );

	outside_cluster_interaction<<<grid, block>>>(
		clusterer.getCenters()->d_data(),
		clusterer.getMasses()->d_data(),
		threads,
		tmp_vel.d_data() );
	CUT_CHECK_ERROR( "kernel launch" );

	threads = planets->size();
	block = min( threads, MIN_THREADS );
	grid = 1 + ( threads - 1 ) / block.x;

	propagate_velocities<<<grid, block>>>(
		tmp_vel.d_data(),
		planets->getPositions().map(MEM::MISC::BUF_CU),
		planets->getVelocities().d_data(),
		clusterer.getShuffle()->d_data(),
		clusterer.getCounts()->d_data(),
		clusterer.getCount() - 1
		);
	CUT_CHECK_ERROR( "kernel launch" );
}

void Phx::CImpl::run_clusters()
{
	if( clusters_on )
	{
		clusterer.kmeans();
	}
}

void Phx::CImpl::update_positions()
{
	unsigned threads = planets->size();
	dim3 block( min( MIN_THREADS, threads ) );
	dim3 grid( 1 + ( threads - 1 ) / block.x );

	update_positions_kernel<<<grid, block>>>(
		planets->getPositions().map(MEM::MISC::BUF_CU),
		planets->getVelocities().d_data(),
		planets->getCount().map(MEM::MISC::BUF_CU) );
	CUT_CHECK_ERROR( "kernel launch" );
}

void Phx::CImpl::handle_collisions()
{
	bool merge_performed = false;
	MEM::MISC::BufferCu<unsigned> merge_needed(1);
	unsigned *in_merges = merges1.d_data();
	unsigned *out_merges = merges2.d_data();

	do
	{
		merge_needed.assign(0);
		unsigned threads = planets->size();
		dim3 block( min( MIN_THREADS, threads ) );
		dim3 grid( 1 + ( threads - 1 ) / block.x );
		
		detect_collisions<<<grid, block>>>(
			planets->getPositions().map(MEM::MISC::BUF_CU),
			planets->getRadiuses().map(MEM::MISC::BUF_CU),
			clusterer.getCounts()->d_data(),
			clusterer.getShuffle()->d_data(),
			clusterer.getCount() - 1,
			out_merges,
			merge_needed.d_data() );
		CUT_CHECK_ERROR("kernel launch");

		if( merge_needed.retrieve() == 0 )
		{
			if( !merge_performed )
				return;

			if( cleaner )
				cleaner->notifyCheckNeeded();
			else
				log_printf( _WARNING, "Merge performed, but cleaner not set!\n" );
			return;
		}
		merge_performed = true;

		MEM::MISC::BufferCu<unsigned> done(1);
		
		do
		{
			done.assign(1);
			std::swap( in_merges, out_merges );
/*                        log_printf(DBG,"grid: %d   block: %d\n",grid.x,block.x);*/
			merge_collisions<<<grid, block>>>(
				in_merges,
				out_merges,
				planets->getPositions().map(MEM::MISC::BUF_CU),
				planets->getVelocities().d_data(),
				planets->getMasses().d_data(),
				planets->getRadiuses().map(MEM::MISC::BUF_CU),
				planets->getCount().map(MEM::MISC::BUF_CU),
				done.d_data() );
			CUT_CHECK_ERROR("kernel launch");
		}
		while( !done.retrieve() );
	}
	while(true);
}

void Phx::CImpl::enableClusters(bool orly)
{
	clusters_on = orly;
}

bool Phx::CImpl::clustersEnabled() const
{
	return clusters_on;
}

void Phx::CImpl::registerCleaner( MEM::MISC::PlanetHolderCleaner *c )
{
	cleaner = c;
}

Phx::Phx(MEM::MISC::PhxPlanetFactory *p)
	: impl( new CImpl(p) )
{
}

Phx::~Phx()
{
	delete impl;
}

void Phx::compute(unsigned n)
{
	impl->compute(n);
}

void Phx::enableClusters(bool orly)
{
	impl->enableClusters(orly);
}

bool Phx::clustersEnabled() const
{
	return impl->clustersEnabled();
}

void Phx::registerCleaner( MEM::MISC::PlanetHolderCleaner *c )
{
	impl->registerCleaner( c );
}
