#ifndef __OPTIONS_H__

#define __OPTIONS_H__

#include <boost/program_options.hpp>
#include <string>

#include "util/config.h"

/** 
 * @brief Klasa odpowiedzialna za stworzenie początkowej konfiguracji
 * programu, na podstawie pliku oraz parametrów linii komend.
 */
class Options {
public:
	/** 
	 * @brief Definiuje opcje konfiguracji programu
	 */
	Options();
	virtual ~Options();

	/** 
	 * @brief Dodaje parametry linii komend do przeczytania.
	 * 
	 * @param argc ilość parametrów
	 * @param argv parametry
	 */
	void addCmdLine( int argc , const char**argv );
	/** 
	 * @brief Dodaje plik konfiguracyjny do przeczytania.
	 * 
	 * @param file ścieżka do pliku.
	 */
	void addCfgFile( const char*file );

	/** 
	 * @brief Zwraca konfigurację początkową programu.
	 * 
	 * @return Konfiguracja.
	 */
	const Config& getCfg();

	/** 
	 * @brief Zwraca string z treścią pomocy programu.
	 * 
	 * @return treść pomocy.
	 */
	std::string getHelp();
	
private:
	bool converted;
	Config cfg;

	boost::program_options::variables_map vm;
	boost::program_options::options_description desc;
};


#endif /* __OPTIONS_H__ */

