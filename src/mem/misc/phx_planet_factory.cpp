#include "phx_planet_factory.h"

#include "cuda/math.h"

using namespace MEM::MISC;

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

float3   PhxPlanet::getPosition() const
{
	ASSERT( exists );
	int id = holder->actualID( login );
	if( id < 0 ) return make_float3(0);
	float3 result = holder->pos.map( BUF_H )[ id ];
	holder->pos.unmap();
	return result;
}

float PhxPlanet::getRadius() const
{
	ASSERT( exists );
	int id = holder->actualID( login );
	if( id < 0 ) return 0.0f;
	float result = holder->radius.map( BUF_H )[ id ];
	holder->radius.unmap();
	return result;
}

float PhxPlanet::getMass() const
{
	ASSERT( exists );
	int id = holder->actualID( login );
	if( id < 0 ) return 0.0f;
	return holder->mass.getAt( id );
}

float3 PhxPlanet::getVelocity() const
{
	ASSERT( exists );
	int id = holder->actualID( login );
	if( id < 0 ) return make_float3(0);
	return holder->velocity.getAt( id );
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

void PhxPlanet::remove()
{
	ASSERT( exists );
	int id = holder->actualID( login );
	if( id < 0 ) return;
	holder->mass.setAt( id, .0f );
	holder->radius.map( BUF_H )[ id ] = .0f; // FIXME; to ssie! kopiuje cały bufor, żeby zmienić jedną wartość :< zresztą z getRadius jest analogiczna buła
	holder->radius.unmap();
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
