#ifndef __DRIVER_H__

#define __DRIVER_H__

#include "SDL/SDL.h"
#include "util/vector.h"

/** 
 * @brief Abstrackja dla kontrolera, takiego jak myszka czy joystick.
 */
class CLocationDriver
{
public:
	virtual ~CLocationDriver() {}
	/** 
	 * @brief Kalibrowanie wskaźnika.
	 * 
	 * @return true jeśli się powiodła, false wpp
	 */
	virtual bool Calibrate() { return true; }
	/** 
	 * @brief Inicjalizacja wskaźnika.
	 * 
	 * @return true jeśli się powiodło, false wpp
	 */
	virtual bool Init() = 0;
	/** 
	 * @brief Zwraca pozycję wskaźnika.
	 * 
	 * @return zwraca 3 wymiary w których może być wskazanie.
	 */
	virtual Vector3 GetPosition() = 0;
	/** 
	 * @brief ustawia pozycje wskaźnika w podanym wektorze.
	 * 
	 * @param ret wektor przez który ma być zwrócony wynik.
	 */
	virtual void GetPosition(Vector3 &ret)
	{	ret = GetPosition(); }
};

/** 
 * @brief Pusty wskaźnik gdy nie ma żadnego innego.
 */
class CNullLD : public CLocationDriver
{
public:
	virtual ~CNullLD() {}
	virtual bool Init();
	virtual Vector3 GetPosition();
};


/** 
 * @brief Implementacja dla joysticka przy pomocy SDLa
 */
class CJoystickLD : public CLocationDriver
{
	SDL_Joystick *mJoystick;
public:
	virtual ~CJoystickLD() {}
	virtual bool Init();
	virtual Vector3 GetPosition();
};

/** 
 * @brief Implementacja zwyczajnej myszki.
 */
class CMouseLD : public CLocationDriver {
public:
	virtual ~CMouseLD() {}
	virtual bool Init();
	virtual Vector3 GetPosition();
};

#endif /* __DRIVER_H__ */

