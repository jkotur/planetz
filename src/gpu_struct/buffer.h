#ifndef BUFFER_H
#define BUFFER_H

namespace GPU
{
	template<class T>
	class Buffer
	{
		public:
			T* h_ptr; // host data pointer
			T* d_ptr; // device data pointer
			size_t size; // size of pointed data == number of elements, not bytes
			// TODO: void fireEventContentChanged();
	};
}

#endif // BUFFER_H
