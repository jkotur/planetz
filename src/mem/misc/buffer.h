#ifndef BUFFER_H
#define BUFFER_H

#include <cstdlib>
#include "debug/routines.h"

namespace MEM
{
namespace MISC
{
	typedef unsigned int uint;
	/** 
	 * @brief Bazowa klasa dla buforów pamięci
	 */
	template<typename T>
	class BufferBase
	{
	public:
		/** 
		 * @brief Konstruktor pustego bufora
		 */
		BufferBase() : length(0), size(0) , realsize(0){}
		virtual ~BufferBase() {}

		/** 
		 * @brief Funkcja zmieniająca wielkość bufora.
		 * 
		 * @param num nowa ilość elementów bufora.
		 * @param data dodatkowo można podać dane które mają być skopiowane do bufora
		 */
		virtual void resize( size_t num , const T*data = NULL ) = 0;

		/**
		 * @brief Pobiera wartość spod konkretnego indeksu.
		 *
		 * @param i Indeks.
		 *
		 * @returns Pobrana wartość.
		 */
		virtual T getAt(unsigned i) const = 0;

		/**
		 * @brief Ustawia wartość pod zadanym indeksem.
		 *
		 * @param i Indeks.
		 *
		 * @param val Wartość do ustawienia.
		 */
		virtual void setAt( unsigned i, const T& val ) = 0;

		/** 
		 * @brief Funkcja przypisująca wartość dla jednoelementowych buforów.
		 * 
		 * @param val wartość do wpisania
		 */
		virtual void assign( T val );

		/**
		 * @brief Pobiera wartość z bufora. Działa tylko dla buforów jednoelementowych.
		 *
		 * @returns Pobrana wartość.
		 */
		virtual T retrieve();

		/** 
		 * @brief Zwraca wielkość bufora.
		 * 
		 * @return wielkość bufora w bajtach
		 */
		virtual size_t getSize() const
		{
			return size;
		}

		/** 
		 * @brief Zwraca długość bufora.
		 * 
		 * @return wielkość bufora w ilości elementów
		 */
		virtual uint   getLen() const
		{
			return length;
		}
	protected:
		/**
		 * @brief Liczba elementów bufora.
		 */
		uint   length;

		/**
		 * @brief Logiczna wielkość (w bajtach) bufora.
		 */
		size_t size;
		
		/**
		 * @brief Fizyczna wielkość (w bajtach) bufora.
		 */
		size_t realsize;
	};

	template<typename T>
	void BufferBase<T>::assign( T val )
	{
		ASSERT_MSG( 1 == getLen(), "assign() works for one field buffers only!" );
		setAt(0, val);
	}

	template<typename T>
	T BufferBase<T>::retrieve()
	{
		ASSERT_MSG( 1 == getLen(), "retrieve() works for one field buffers only!" );
		return getAt(0);
	}
}
}
#endif // BUFFER_H

