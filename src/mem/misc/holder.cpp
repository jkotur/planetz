#include "holder.h"

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

