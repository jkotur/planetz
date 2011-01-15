#ifndef __MEM_MISC_COMPACTER_H__
#define __MEM_MISC_COMPACTER_H__
#include "buffer_cu.hpp"
#include "cudpp.h"
#include <map>
#include <list>

namespace MEM
{
namespace MISC
{
/**
 * @brief Klasa pakująca bufory wg zadanej maski.
 *
 * @todo W tej chwili klasa zakłada, że dostanie przynajmniej jeden bufor o
 * romiarze elementu równemu sizeof(unsigned int), oraz że największy rozmiar
 * nie będzie większy niż jego trzykrotność - sizeof(float3). W razie zmiany
 * użycia zwrócić na to uwagę.
 */
class Compacter
{
	public:
		/**
		 * @brief Inicjalizacja.
		 *
		 * @param _mask Maska, wg której mają zostać filtrowane bufory.
		 */
		Compacter( BufferCu<unsigned> *_mask );
		virtual ~Compacter();
		
		/**
		 * @brief Dodaje bufor do filtrowania.
		 *
		 * @param d_data Adres w pamięci karty.
		 *
		 * @param elem_size Rozmiar jednego elementu d_data.
		 */
		void add( void *d_data, unsigned elem_size );

		/**
		 * @brief Kontener służący do wymiany informacji o zmianach indeksów.
		 *
		 * @note Używanie go do sprawdzania dużej ilości indeksów nie jest
		 * zbyt optymalne. Nie jest też prawdopodobnie potrzebne.
		 */
		typedef std::map<const unsigned, int> IdxChangeSet;

		/**
		 * @brief Dokonuje kompresji na dodanych buforach.
		 *
		 * @param idx_change_set Opcjonalny parametr, używany do notyfikacji
		 * o zmianach indeksów. Mapa, której klucze są indeksami, które nas
		 * interesują. Odpowiadające im wartości to nowe indeksy lub -1,
		 * jeżeli obiekt został usunięty spod zadanego indeksu.
		 *
		 * @returns Nowy rozmiar buforów.
		 */
		size_t compact( IdxChangeSet *idx_change_set=NULL );

	private:
		typedef std::list<void*> PtrList;
		typedef std::map<unsigned, PtrList> PtrListMap;

		void createScanHandle();
		unsigned compactLoop( BufferCu<unsigned> *mask, BufferCu<unsigned> *indices, const PtrList& list );
		void scan( BufferCu<unsigned> *in, BufferCu<unsigned> *out );
		void updateIndices( BufferCu<unsigned> *indices, IdxChangeSet *idx_change_set );

		CUDPPHandle scanHandle;
		unsigned size;
		BufferCu<unsigned> *mask;

		PtrListMap map;
}; // Compacter
} // MISC
} // MEM

#endif // __MEM_MISC_COMPACTER_H__

