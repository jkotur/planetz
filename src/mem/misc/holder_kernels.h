#include "holder.h"

namespace MEM
{
namespace MISC
{
	/**
	 * @brief "Rozciąga" bufor zadaną ilość razy.
	 *
	 * @detail W nowym buforze każda z wartości bufora poprzedniego zapisana jest factor razy.
	 *
	 * @param in Bufor do rozciągnięcia.
	 *
	 * @param out Bufor wynikowy.
	 *
	 * @param factor Mnożnik. Bufor wynikowy jest factor razy większy od wejściowego.
	 */
	void stretch( MEM::MISC::BufferCu<unsigned> *in, MEM::MISC::BufferCu<unsigned> *out, unsigned factor );
}
}
