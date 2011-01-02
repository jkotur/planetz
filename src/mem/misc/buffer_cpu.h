#ifndef _BUFFER_CPU_H_
#define _BUFFER_CPU_H_

#include <vector>
#include "buffer.h"

namespace MEM
{
namespace MISC
{
	/// @brief Implementacja bufora przechowującego dane wyłącznie na CPU.
	template<typename T>
	class BufferCpu
		: public BufferBase<T>
		, public std::vector<T>
	{
	public:
		/// @brief Tworzy bufor o podanym rozmiarze.
		/// @param num Rozmiar bufora.
		BufferCpu( unsigned num = 0 )
		{
			resize( num );
		}

		virtual void resize( size_t num, const T*data = NULL )
		{
			std::vector<T>::resize( num );
			if( data )
			{
				std::vector<T>::assign( data, data + num );
			}
		}

		virtual void assign( T val )
		{
			std::vector<T>::assign( &val, (&val) + 1 );
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
