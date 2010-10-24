#ifndef __LAYOUT_H__

#define __LAYOUT_H__

#include <string>
#include <CEGUI.h>

/**
 * Klasa layoutu.
 * UWAGA! klasa rejestruje głowne okno
 * menagera okien, więc działać może 
 * poprawnie tylko jedna instancja klasy
 */
class Layout {
public:
	Layout ( const std::string& name );
	virtual ~Layout();
	
private:
	
};


#endif /* __LAYOUT_H__ */

