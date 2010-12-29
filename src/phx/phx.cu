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
		void run_clusters();

		MEM::MISC::PhxPlanetFactory *planets;
		Clusterer clusterer;

		MEM::MISC::BufferCu<float3> tmp_pos;
		MEM::MISC::BufferCu<float3> tmp_vel;
		MEM::MISC::BufferCu<float> tmp_mass;

		bool clusters_on;
};

Phx::CImpl::CImpl(MEM::MISC::PhxPlanetFactory *p)
	: planets(p)
	, clusterer( &p->getPositions(), &p->getMasses() )
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
	}
	unmap_buffers();
}

void Phx::CImpl::map_buffers()
{
	planets->getPositions().map( MEM::MISC::BUF_CU );
	planets->getRadiuses().map( MEM::MISC::BUF_CU );
	planets->getCount().map( MEM::MISC::BUF_CU );
	tmp_pos.resize( planets->size() );
	tmp_vel.resize( planets->size() );
	cudaMemset( tmp_pos.d_data(), 0, planets->size() * sizeof(float3) );
	cudaMemset( tmp_vel.d_data(), 0, planets->size() * sizeof(float3) );
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
#ifdef PHX_DEBUG
		log_printf( DBG, "counts[%u]: %u\n", c, h_counts[c] );
#endif
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
			c, // cluster id
			tmp_pos.d_data(),
			tmp_vel.d_data()
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
	cudaMemcpy(
		planets->getPositions().map(MEM::MISC::BUF_CU),
		tmp_pos.d_data(), planets->size() * sizeof(float3), cudaMemcpyDeviceToDevice );
	cudaMemcpy(
		planets->getVelocities().d_data(),
		tmp_vel.d_data(), planets->size() * sizeof(float3), cudaMemcpyDeviceToDevice );
#ifdef PHX_DEBUG
//	PRINT_OUT_BUF( whois, "%u" );
#endif
	delete[] h_counts;
}

void Phx::CImpl::run_nbodies( unsigned threads )
{	
	ASSERT( threads );
	if( clusters_on )
	{
		run_nbodies2();
		return;
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
		planets->getCount().map(MEM::MISC::BUF_CU),
		tmp_pos.d_data(),
		tmp_vel.d_data()
#ifdef PHX_DEBUG
		, d_dvs, 4210
#endif
		);
	
	CUT_CHECK_ERROR("Kernel launch");
	
	
	cudaMemcpy( planets->getPositions().map(MEM::MISC::BUF_CU), tmp_pos.d_data(), threads * sizeof(float3), cudaMemcpyDeviceToDevice );
	cudaMemcpy( planets->getVelocities().d_data(), tmp_vel.d_data(), threads * sizeof(float3), cudaMemcpyDeviceToDevice );
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

void Phx::CImpl::run_clusters()
{
	if( clusters_on )
	{
		clusterer.kmeans();
	}
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
