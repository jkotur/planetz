#ifndef __CONFIG_H__

#define __CONFIG_H__

#include <map>
#include <string>

#include <boost/any.hpp>

/** 
 * @brief Klasa zawierająca informację o konfiguracji aplikacji
 */
class Config {
public:
	/** 
	 * @brief Konstruuje pustą konfigurację
	 */
	Config ()
	{
	}

	/** 
	 * @brief Sprzątak konfigurację
	 */
	virtual ~Config()
	{
	}

	/** 
	 * @brief Ustawia zadane pole konfiguracji na zadaną wartość o zadanym typie
	 * 
	 * @param key nazwa ustawianego pola
	 * @param val wartość ustawianego pola
	 */
	template<typename T>
	void set( const std::string& key , T val )
	{
		map[key] = boost::any( val );
	}

	/** 
	 * @brief Zwraca wartość interesującego pola. Jeśli pole nie zostanie odnalezione,
	 * Bądź jego typ różni się od typu zawartego w konfiguracji, zwórci domyślną
	 * wartość dla danego typu (domyślny konstruktor).
	 * 
	 * @param key nazwa interesującego pola
	 * 
	 * @return wartość pola.
	 */
	template<typename T>
	T get( const std::string& key ) const
	{
		try {
			std::map<std::string,boost::any>::const_iterator i = map.find(key);
			if( i == map.end() )
				return T();
			else	return boost::any_cast<T>(i->second);
		} catch( const boost::bad_any_cast& ) {
			return T();
		}
	}

	/** 
	 * @brief Czyści konfigurację do pustej.
	 */
	void clear() { map.clear(); }
private:
	std::map<std::string,boost::any> map;
};


#endif /* __CONFIG_H__ */

