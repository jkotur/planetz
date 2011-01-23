
#ifndef __UI_H__

#define __UI_H__

#include <list>

#include <boost/signals.hpp>
#include <boost/bind.hpp>

#include "input/driver.h"

#include "gfx/drawable.h"

#include "./gui/gui.h"
#include "./camera.h"
#include "./planetz_setter.h"
#include "./gui/layout.h"

#include "ui/input_listener.h"

namespace UI
{
/** 
 * @brief Klasa odpowiedzialna za wszystko co związane z interfejsem użytkownika.
 * Odpowiada zarówno za peryferia takie jak myszka/klawiatura/joystick,
 * ale również za zmianę wielkości okna, oraz wyświetlanie graficznego 
 * interfejsu użytkonika.
 */
class UI : public GFX::Drawable {
	struct breaker {                                                                       
		typedef void result_type;                                        

		template<typename InputIterator>                                
		result_type operator()(InputIterator first, InputIterator last) const   
		{
			while( first != last && *first == false ) ++first;
		}
	};
public:
	UI ();
	virtual ~UI();

	/** 
	 * @brief Inicjalizuję klasę, by była gotowa do działania.
	 * 
	 * @return true jeśli inicjalizacja się powiedzie, false wpp
	 */
	bool init();

	/** 
	 * @brief Wyświetla GUI na ekran
	 */
	virtual void draw() const;

	/** 
	 * @brief Ustawia layout gui
	 * 
	 * @param layout
	 */
	void set_layout( Layout*layout );

	/** 
	 * @brief Informuję ui że jedna klatka została wyświetlona
	 */
	void signal();

	/** 
	 * @brief Powoduje zczytanie kolejki zdarzeń które wystąpiły do tej pory.
	 * 
	 * @return zwraca 0 jeśli program powinien zakończyć działanie, 1 wpp
	 */
	int event_handle();

	/** 
	 * @brief Funkcja dodająca słuchacza wejścia programu.
	 * 
	 * @param lst słuchacz wejścia
	 * @param level poziom na którym powinien być przyłączony sygnał
	 */
	void add_listener( InputListener* lst , int level = 0 );

	/** 
	 * @brief Funkcja usuwająca słuchacza wejścia programu
	 * 
	 * @param lst słuchacz wejścia
	 */
	void del_listener( InputListener* lst );
	
	/** @brief Sygnał emitowany gdy klawisz zostaje zwolniony */
	boost::signal<void (SDLKey,Uint16,Uint8)> sigKeyUp;
	/** @brief Sygnał emitowany gdy klawisz zostaje wciśnięty */
	boost::signal<void (SDLKey,Uint16,Uint8)> sigKeyDown;
	/** @brief Sygnał emitowany gdy myszka zmieni pozycję */
	boost::signal<void (int,int)> sigMouseMotion;
	/** @brief Sygnał emitowany gdy guzik myszki zostaje zwolniony */
	boost::signal<void (int,int,int)> sigMouseButtonUp;
	/** @brief Sygnał emitowany gdy guzik myszki zostaje wciśnięty */
	boost::signal<bool (int,int,int) , breaker > sigMouseButtonDown;
	/** @brief Sygnał emitowany gdy zmienia się wielkość ekranu */
	boost::signal<void (int,int)> sigVideoResize;

private:
	Gui gui;
	CLocationDriver*joy;
};

}; // UI

#endif /* __UI_H__ */

