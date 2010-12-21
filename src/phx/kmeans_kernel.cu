// 2 * k < elems_size
__global__ void kmeans__findbest_kernel(unsigned k, float3* means, float3* elems, unsigned elems_size, unsigned* assignments, float* errors)
{
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ float3 s_means[];
	float x = elems[index].x;
	float y = elems[index].y;
	float z = elems[index].z;

	if( index >= elems_size )
		return;
	if( threadIdx.x < k )//@todo k > 512
		s_means[threadIdx.x] = means[threadIdx.x];

	__syncthreads();
	
	///@todo pozbyć się konfliktów w bankach pamięci?
	unsigned best = 0;
	float dx = s_means[0].x - x;
	float dy = s_means[0].y - y;
	float dz = s_means[0].z - z;
	float dist = dx * dx + dy * dy + dz * dz;
	float best_dist = dist;

	for(unsigned i = 1; i < k; ++i)
	{
		dx = s_means[i].x - x;
		dy = s_means[i].y - y;
		dz = s_means[i].z - z;
		dist = dx * dx + dy * dy + dz * dz;
		if(dist < best_dist)
		{
			best_dist = dist;
			best = i;
		}
	}
	assignments[ index ] = best;
	errors[ index ] = best_dist;
}

__global__ void kmeans__prepare_kernel(unsigned* to_sort, unsigned num)
{
	unsigned index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < num)
		to_sort[index] = index;
}

///Liczy ilości wystąpień wartości nie większych od danej w posortowanej tablicy
__global__ void kmeans__count_kernel(unsigned *assignments, unsigned *counts, unsigned num_minus_1, unsigned k)
{
	unsigned index = threadIdx.x + blockIdx.x * blockDim.x;
	if( index > num_minus_1 )
		return;
	unsigned prev = assignments[ index ];
	unsigned next;
	if( index != num_minus_1 )
		next = assignments[ index + 1 ];
	else
		next = k;
	while( prev != next )
		counts[ prev++ ] = index + 1;
}

__device__ float3& operator+=(float3& a, const float3& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

__device__ float3 operator+(const float3&a, const float3&b)
{
	return make_float3( a.x + b.x, a.y + b.y, a.z + b.z );
}

__device__ float3 operator/(const float3& a, const float& b)
{
	return make_float3( a.x / b, a.y / b, a.z / b );
}

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

template <unsigned int blockSize>
__global__ void reduceSelective(float3 *g_idata, float3 *g_odata, unsigned *counts, unsigned id, unsigned* shuffle)
{
	extern __shared__ float3 s_data[];
	float3 *sdata = s_data; // WTF!? inaczej nie działa - wyjebka kompilacji
	unsigned int n = counts[id];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid + (id ? counts[id-1] : 0);
	unsigned int gridSize = blockSize*gridDim.x;
	sdata[tid] = make_float3(0,0,0);
	while (i < n)
	{
		sdata[tid] += g_idata[ shuffle[i] ];
		i += gridSize;
	}
	__syncthreads();
	reduce<float3, blockSize>(g_idata, n, tid, i, sdata);
	if (tid == 0) g_odata[id] = sdata[0] / (n - (id ? counts[id-1] : 0));
}

template <class T, unsigned int blockSize>
__global__ void reduceFull(T *g_idata, T *g_odata, unsigned n)
{
	extern __shared__ T sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*gridDim.x;
	sdata[tid] = (T)0;
	while (i < n) { sdata[tid] += g_idata[ i ]; i += gridSize; }
	__syncthreads();
	reduce<T, blockSize>(g_idata, n, tid, i, sdata);
}
