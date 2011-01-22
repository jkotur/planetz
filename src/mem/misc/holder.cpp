#include "holder.h"
#include "compacter.h"

using namespace MEM::MISC;

ClusterHolder::ClusterHolder()
	: m_size(0)
{
}

ClusterHolder::~ClusterHolder()
{
}

void ClusterHolder::resize(size_t k_size, size_t n_size)
{
	centers.resize(k_size);
	masses.resize(k_size);
	assignments.resize(n_size);
	m_size = k_size;
}

size_t ClusterHolder::k_size() const
{
	return m_size;
}

namespace MEM
{
namespace MISC
{
unsigned __filter( PlanetHolder *what, BufferCu<unsigned> *how, IdxChangeSet *changes )
{
	Compacter c( how );
	c.add( what->model.map(BUF_CU), sizeof(int) );
	c.add( what->texId.map(BUF_CU), sizeof(int) );
	c.add( what->light.map(BUF_CU), sizeof(float) );
	c.add( what->radius.map(BUF_CU), sizeof(float) );
	c.add( what->mass.d_data(), sizeof(float) );
	c.add( what->atm_data.map(BUF_CU), sizeof(float2) );
	c.add( what->atm_color.map(BUF_CU), sizeof(float3) );
	c.add( what->pos.map(BUF_CU), sizeof(float3) );
	c.add( what->velocity.d_data(), sizeof(float3) );
	//c.add( what->color.map(BUF_CU), sizeof(float4) );
	
	unsigned newSize = c.compact( changes );
	cudaMemcpy( what->count.map(BUF_CU), &newSize, sizeof(unsigned), cudaMemcpyHostToDevice );

	//what->color.unmap();
	what->count.unmap();
	what->model.unmap();
	what->texId.unmap();
	what->light.unmap();
	what->radius.unmap();
	what->atm_data.unmap();
	what->atm_color.unmap();
	what->pos.unmap();

	return newSize;
}
}
}

