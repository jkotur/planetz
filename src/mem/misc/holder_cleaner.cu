#include "holder_cleaner.h"
#include "holder_cleaner_kernels.h"

using namespace MEM::MISC;

PlanetHolderCleaner::PlanetHolderCleaner( PhxPlanetFactory *f )
	: fact( f )
	, planetsInUse( 1 )
	, needChecking( false )
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

void PlanetHolderCleaner::notifyCheckNeeded()
{
	needChecking = true;
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
	unsigned threads = fact->size();
	// block.x i argument template'a muszą się zgadzać - przydałby się jakiś switch - najlepiej ładnie opakować redukcję
	dim3 block( 512 );//min( 512, threads ) );
	dim3 grid( 1 );
	reduceFull<unsigned, 512><<<grid, block>>>(
		filter.d_data(),
		planetsInUse.d_data(),
		threads );
	//log_printf( INFO, "%u of %u planets in use.\n", planetsInUse.retrieve(), threads );
	return planetsInUse.retrieve() < 0.8 * threads - 20; // magic numbers!
}

void PlanetHolderCleaner::filterHolder()
{
	fact->filter( &filter );
}
