#include <limits>
#include "phx.h"
#include "phx_kernels.h"
#include "phx_templates.h"
#include "kmeans.h"
#include "cuda/math.h"

using namespace PHX;

ConstChecker<float3, MEM::MISC::BufferGl> pos_checker;
ConstChecker<float, MEM::MISC::BufferCu> mass_checker;
ConstChecker<float3, MEM::MISC::BufferCu> vel_checker;

class Phx::CImpl
{
	public:
		CImpl(MEM::MISC::PhxPlanetFactory *p);
		virtual ~CImpl();

		void compute(unsigned n);
		void enableClusters(bool orly);
		bool clustersEnabled() const;

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
		MEM::MISC::BufferCu<unsigned> filter; // a może użyć jakiegoś merge'a? taniej dla ramu
		bool clusters_on;
};

Phx::CImpl::CImpl(MEM::MISC::PhxPlanetFactory *p)
	: planets( p )
	, clusterer( p )
	, clusters_on( true )
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
		vel_checker.setBuf( &planets->getVelocities() );
		mass_checker.setBuf( &planets->getMasses() );
		pos_checker.setBuf( &planets->getPositions() );
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
		dim3 block( min( threads, 512 ) );
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
	dim3 block( min( threads, 512 ) );
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
	ASSERT( threads <= 512 );
	dim3 block( min( threads, 512 ) );
	dim3 grid( 1 );

	static unsigned print_modulo = 0;
	print_modulo = (print_modulo+1)%5000;
	MEM::MISC::BufferCu<float3> *centers = clusterer.getCenters();
	centers->bind();
	if( print_modulo == 0 )
	for( unsigned i = 0; i < centers->getLen(); ++i )
	{
		log_printf( DBG, "centers[%u] = (%f,%f,%f)\n", i, centers->h_data()[i].x, centers->h_data()[i].y, centers->h_data()[i].z );
	}
	centers->unbind();
	outside_cluster_interaction<<<grid, block>>>(
		clusterer.getCenters()->d_data(),
		clusterer.getMasses()->d_data(),
		threads,
		tmp_vel.d_data() );
	CUT_CHECK_ERROR( "kernel launch" );

	tmp_vel.bind();
	if( print_modulo == 0 )
	for( unsigned i = 0; i < centers->getLen(); ++i )
	{
		log_printf( DBG, "vel[%u] = (%f, %f, %f)\n", i, tmp_vel.h_data()[i].x, tmp_vel.h_data()[i].y, tmp_vel.h_data()[i].z );
	}
	tmp_vel.unbind();

	threads = planets->size();
	block = min( threads, 512 );
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
	dim3 block( min( 512, threads ) );
	dim3 grid( 1 + ( threads - 1 ) / block.x );

	update_positions_kernel<<<grid, block>>>(
		planets->getPositions().map(MEM::MISC::BUF_CU),
		planets->getVelocities().d_data(),
		planets->getCount().map(MEM::MISC::BUF_CU) );
	CUT_CHECK_ERROR( "kernel launch" );
}

void Phx::CImpl::handle_collisions()
{
	bool merge_was_needed = false;
	MEM::MISC::BufferCu<unsigned> merge_needed(1);
	unsigned *in_merges = merges1.d_data();
	unsigned *out_merges = merges2.d_data();

	do
	{
		merge_needed.assign(0);
		unsigned threads = planets->size();
		dim3 block( min( 512, threads ) );
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
			if( merge_was_needed )
			{
				TODO("rzadziej filtrować");
				filter.resize(threads);
				create_filter<<<grid, block>>>(
					planets->getMasses().d_data(),
					filter.d_data(),
					planets->getCount().map(MEM::MISC::BUF_CU) );
				planets->filter( &filter );
			}
			return;
		}
		merge_was_needed = true;

	/*	merges2.bind();
		for( unsigned i = 0; i < threads; ++i )
		{
			if( merges2.h_data()[i] != i )
				log_printf( DBG, "merges[%u] = %u\n", i, merges2.h_data()[i] );
		}
		merges2.unbind();
		static unsigned safety_buf = 0;
		if( ++safety_buf > 10 ) abort(); */
		MEM::MISC::BufferCu<unsigned> done(1);
		
		unsigned __debug_counter = 0;
		do
		{
			log_printf(DBG, "mergin' #%u\n", __debug_counter++ );
			if( __debug_counter >= threads )
			{
				PRINT_OUT_BUF( merges1, "%u" );
				PRINT_OUT_BUF( merges2, "%u" );
				abort();
			}
			done.assign(1);
			std::swap( in_merges, out_merges );
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
