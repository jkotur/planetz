#ifndef __CONFIG_H__

#define __CONFIG_H__

#include <map>
#include <string>

#include <boost/any.hpp>

class Config {
public:
	Config ()
	{
	}

	virtual ~Config()
	{
	}

	template<typename T>
	void set( const std::string& key , T val )
	{
		map[key] = boost::any( val );
	}

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

	void clear() { map.clear(); }
private:
	std::map<std::string,boost::any> map;
};


#endif /* __CONFIG_H__ */

