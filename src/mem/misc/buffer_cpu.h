#ifndef _BUFFER_CPU_H_
#define _BUFFER_CPU_H_

#include <vector>
#include "buffer.h"

namespace MEM
{
namespace MISC
{
	template<typename T>
	class BufferCpu
		: public BufferBase<T>
		, public std::vector<T>
	{
		virtual void resize( size_t num, const T*data = NULL )
		{
			std::vector<T>::resize( num );
			std::vector<T>::assign( data, data + num );
		}

		virtual size_t getSize() const
		{
			return std::vector<T>::size() * sizeof(T);
		}

		virtual uint getLen() const
		{
			return std::vector<T>::size();
		}
	};
} // namespace MEM::MISC
} // namespace MEM

#endif // _BUFFER_CPU_H_
