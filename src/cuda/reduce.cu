template <class T, unsigned int blockSize>
__device__ void reduce(T *g_idata, unsigned n, unsigned tid, unsigned i, T sdata[])
{
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32)
	{
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
}

template <class T, unsigned int blockSize>
__global__ void reduceFull(T *g_idata, T *g_odata, unsigned n)
{
	__shared__ T sdata[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*gridDim.x;
	sdata[tid] = (T)0;
	while (i < n) { sdata[tid] += g_idata[ i ]; i += gridSize; }
	__syncthreads();
	reduce<T, blockSize>(g_idata, n, tid, i, sdata);
	if( tid == 0 ) g_odata[blockIdx.x] = sdata[0];
}

