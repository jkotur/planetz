#include "phx_planet_factory.h"

#include "cuda/math.h"

using namespace MEM::MISC;

#define PHX_PLANET_GET_IMPL( buf, default_val )	\
	ASSERT( exists );			\
	int id = holder->actualID( login );	\
	if( id < 0 ) return default_val;	\
	return buf.getAt( id );

PhxPlanet::PhxPlanet()
	: exists( false )
{
}

PhxPlanet::PhxPlanet( unsigned id , PlanetHolder* h )
	: login( h->createLogin( id ) )
	, holder( h )
	, exists( true )
{
	ASSERT( h );
}

PhxPlanet::PhxPlanet( const PhxPlanet& other )
{
	initFromOther( other );
}

PhxPlanet::~PhxPlanet()
{
	if( exists )
	{
		holder->releaseLogin( login );
	}
}

int PhxPlanet::getId() const
{
	ASSERT( exists );
	return holder->actualID( login );
}

float3 PhxPlanet::getPosition() const
{
	PHX_PLANET_GET_IMPL( holder->pos, make_float3(0) );
}

float PhxPlanet::getRadius() const
{
	PHX_PLANET_GET_IMPL( holder->radius, .0f );
}

float PhxPlanet::getMass() const
{
	PHX_PLANET_GET_IMPL( holder->mass, .0f );
}

float3 PhxPlanet::getVelocity() const
{
	PHX_PLANET_GET_IMPL( holder->velocity, make_float3(0) );
}

void PhxPlanet::initFromOther( const PhxPlanet& other )
{
	holder = other.holder;
	exists = other.exists;
	if( exists && -1 != other.getId() )
	{
		login = holder->createLogin( other.getId() );
	}
}

PhxPlanet& PhxPlanet::operator=( const PhxPlanet& rhs )
{
	if( this == &rhs )
	{
		return *this;
	}
	if( exists )
	{
		holder->releaseLogin( login );
	}
	initFromOther( rhs );
	return *this;
}

bool PhxPlanet::isValid() const
{
	if( !exists ) return false; // niezainicjalizowana
	if( -1 == getId() ) return false; // fizycznie usunięta
	if( .0f == getMass() ) return false; // logicznie usunięta
	return true;
}

PhxPlanetFactory::PhxPlanetFactory( PlanetHolder* holder )
	: holder(holder)
{
}

PhxPlanetFactory::~PhxPlanetFactory( )
{
}

PhxPlanet PhxPlanetFactory::getPlanet( int id )
{
	return PhxPlanet( id , holder );
}

BufferGl<float3>  &PhxPlanetFactory::getPositions()
{
	return holder->pos;
}

BufferGl<float>   &PhxPlanetFactory::getRadiuses()
{
	return holder->radius;
}

BufferGl<uint32_t>&PhxPlanetFactory::getCount()
{
	return holder->count;
}

BufferCu<float> &PhxPlanetFactory::getMasses()
{
	return holder->mass;
}

BufferCu<float3> &PhxPlanetFactory::getVelocities()
{
	return holder->velocity;
}

unsigned PhxPlanetFactory::size() const
{
	return holder->size();
}

void PhxPlanetFactory::filter( BufferCu<unsigned> *mask )
{
	holder->filter( mask );
}
