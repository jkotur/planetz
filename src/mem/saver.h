#ifndef __SAVER_H__

#define __SAVER_H__

#include <string>

namespace MEM
{
namespace MISC
{
	class SaverParams;
}
	/**
	 * @brief Klasa odpowiedzialna za zapis symulacji do pliku.
	 */
	class Saver
	{
	public:
		Saver();
		virtual ~Saver();

		/**
		 * @brief Zapisuje symulację.
		 *
		 * @param source Parametry symulacji, które należy zapisać.
		 *
		 * @param path Ścieżka do pliku, w którym ma zostać zapisana symulacja.
		 */
		void save( const MISC::SaverParams *source, const std::string& path );
		/**
		 * @brief Wczytuje symulację.
		 *
		 * @param dest Miejsce na parametry symulacji, które zostaną wczytane.
		 *
		 * @param path Ścieżka do pliku, z którego ma zostać wczytana symulacja.
		 */
		void load( MISC::SaverParams *dest, const std::string& path );
	};
}

#endif /* __SAVER_H__ */

