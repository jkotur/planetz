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

		/** 
		 * @brief zmienia wielkość bufora
		 * 
		 * @param num nowa ilość elementów bufora
		 * @param preserve_data nieużywana w tej wersji bufora. Dane zawsze są zachowywane
		 * @param data nowe dane do skopiowania
		 */
		virtual void resize( size_t num , bool preserve_data = true , const T*data = NULL )
		{
			std::vector<T>::resize( num );
			if( data )
			{
				std::vector<T>::assign( data, data + num );
			}
		}

		virtual void setAt( unsigned i, const T& val )
		{
			std::vector<T>::at( i ) = val;
		} 
		
		virtual T getAt( unsigned i ) const
		{
			return std::vector<T>::at( i );
		}

		virtual size_t getSize() const
		{
			return std::vector<T>::size() * sizeof(T);
		}

		virtual uint getLen() const
		{
			return std::vector<T>::size();
		}

		virtual void assign( T val )
		{
			return BufferBase<T>::assign( val ); // to resolve ambiguity
		}
	};
} // namespace MEM::MISC
} // namespace MEM

#endif // _BUFFER_CPU_H_
