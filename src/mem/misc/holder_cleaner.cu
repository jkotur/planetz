#include "holder_cleaner.h"
#include "holder_cleaner_kernels.h"

using namespace MEM::MISC;

PlanetHolderCleaner::PlanetHolderCleaner( PhxPlanetFactory *f, FilteringPolicy p )
	: fact( f )
	, planetsInUse( 1 )
	, needChecking( false )
	, filteringPolicy( p )
{
}

PlanetHolderCleaner::~PlanetHolderCleaner()
{
}

void PlanetHolderCleaner::work()
{
	if( !needChecking )
	{
		return;
	}
	needChecking = false;
	
	createFilter();
	if( filteringNeeded() )
	{
		filterHolder();
	}
}

void PlanetHolderCleaner::forceFilter()
{
	if( 0 == fact->size() )
		return;
	createFilter();
	filterHolder();
	needChecking = false;
}

void PlanetHolderCleaner::notifyCheckNeeded()
{
	needChecking = true;
}

void PlanetHolderCleaner::setFilteringPolicy( FilteringPolicy p )
{
	filteringPolicy = p;
}

void PlanetHolderCleaner::createFilter()
{
	unsigned threads = fact->size();
	dim3 block( min( 512, threads ) );
	dim3 grid( 1 + ( threads - 1 ) / block.x );
	filter.resize(threads);
	create_filter<<<grid, block>>>(
		fact->getMasses().d_data(),
		filter.d_data(),
		threads );
}

bool PlanetHolderCleaner::filteringNeeded()
{
	if( Always == filteringPolicy ) return true;
	if( Never == filteringPolicy ) return false;
	unsigned threads = fact->size();
	//TODO block.x i argument template'a muszą się zgadzać - przydałby się jakiś switch - najlepiej ładnie opakować redukcję
	dim3 block( 512 );//min( 512, threads ) );
	dim3 grid( 1 );
	reduceFull<unsigned, 512><<<grid, block>>>(
		filter.d_data(),
		planetsInUse.d_data(),
		threads );
	//log_printf( INFO, "%u of %u planets in use.\n", planetsInUse.retrieve(), threads );
	if( Frequently == filteringPolicy )
		return planetsInUse.retrieve() + 20 < threads;
	ASSERT( Rarely == filteringPolicy );
	return planetsInUse.retrieve() + 20 < 0.8 * threads; // magic numbers!
}

void PlanetHolderCleaner::filterHolder()
{
	fact->filter( &filter );
}

