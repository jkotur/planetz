#ifndef __INPUT_LISTENER_H__

#define __INPUT_LISTENER_H__

#include <SDL/SDL.h>

namespace UI
{

class InputListener {
protected:
	InputListener () {}
	virtual ~InputListener() {}
public:
	
	/** 
	 * @brief Funkcja reagująca na wyciśnięcie klawisza
	 * 
	 * @param SDLKey Wartość enuma klawisza
	 * @param Uint16 Wartość utf16 dla tego klawisza
	 * @param Uint8 Wartość kodu klawiatury
	 */
	virtual void on_key_up( SDLKey , Uint16 , Uint8 ) {}
	/** 
	 * @brief Funkcja reagująca na wciśnięcie klawisza.
	 * 
	 * @param SDLKey Wartość enuma klawisza
	 * @param Uint16 Wartość utf16 dla tego klawisza
	 * @param Uint8 Wartość kodu klawiatury
	 */
	virtual void on_key_down( SDLKey , Uint16 , Uint8 ) {}
	/** 
	 * @brief Funkcja reagująca na ruch myszki.
	 * 
	 * @param x nowa pozycja myszki w osi OX
	 * @param y nowa pozycja myszki w osi OY
	 */
	virtual void on_mouse_motion( int x , int y ) {}
	/** 
	 * @brief Funkcja reagująca na odciśnięcie guzika myszki.
	 * 
	 * @param b kod guzika
	 * @param x pozycja gdzie nastąpiło odciśnięcie guzika w osi OX
	 * @param y pozycja gdzie nastąpiło odciśnięcie guzika w osi OY
	 */
	virtual void on_button_up( int , int , int ) {}
	/** 
	 * @brief Funkcja reagująca na wciśnięcie guzika myszki.
	 * 
	 * @param b kod guzika
	 * @param x pozycja gdzie nastąpiło wciśnięcie guzika w osi OX
	 * @param y pozycja gdzie nastąpiło wciśnięcie guzika w osi OY
	 * 
	 * @return zawsze zwraca false (nie blokuje wywołania kolejnych sygnałów).
	 */
	virtual bool on_button_down( int , int , int ) { return false; }
	/** 
	 * @brief Funkcja powinna być wywoływana co klatkę animacji.
	 */
	virtual void signal() {}
};

}; // UI


#endif /* __INPUT_LISTENER_H__ */

