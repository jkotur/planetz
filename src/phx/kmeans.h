#pragma once
#include <GL/glew.h>
#include <GL/gl.h>

#include "mem/misc/buffer.h"
#include "mem/misc/buffer_cu.hpp"

const float EPSILON = 1e-5;

class PointSet
{
	public:
		PointSet();
		~PointSet();
		void randomize();
		void kmeans( unsigned k );
		void setBounds( unsigned x , unsigned y )
			{ width = x; height = y; }
	//	BufferGl* getBuffer()
	//		{ return buf; }
	private:
		void randomize(float3* d_t, unsigned size);
		void sortByCluster();
		void reduceMeans(float3*, unsigned);
		float reduceErrors();
	/*	BufferCu<unsigned>* shuffle;
		BufferCu<unsigned>* assignments;
		BufferCu<unsigned>* counts;
		BufferCu<float3>* means;
		BufferCu<float>* errors;
		BufferGl* buf;
	*/	unsigned width, height;
};
