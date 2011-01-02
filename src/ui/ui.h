
#ifndef __UI_H__

#define __UI_H__

#include <boost/signals.hpp>
#include <boost/bind.hpp>

#include "input/driver.h"

#include "gfx/drawable.h"

#include "./gui/gui.h"
#include "./camera.h"

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
	 * @brief Informuję ui że jedna klatka została wyświetlona
	 */
	void signal();

	/** 
	 * @brief Powoduje zczytanie kolejki zdarzeń które wystąpiły do tej pory.
	 * 
	 * @return zwraca 0 jeśli program powinien zakończyć działanie, 1 wpp
	 */
	int event_handle();
	
	/** @brief Sygnał emitowany gdy klawisz zostaje zwolniony */
	boost::signal<void (int,int,int)> sigKeyUp;
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

	Gui gui;
private:
	CLocationDriver*joy;
};

#endif /* __UI_H__ */

