#ifndef __WINDOW_H__

#define __WINDOW_H__

#include <SDL/SDL.h>

#include <vector>

/** @brief Klasa odpowiedzialan za obsłógę okna aplikacji.  */
class Window {
public:
	/** 
	 * @brief Tworzy nowe okno.
	 * 
	 * @param res wektor zawierający dwa wymiary okna
	 * @param fs jeśli true tryb pełnoekranowy jest włączony
	 */
	Window( const std::vector<int>& res , bool fs );
	/** 
	 * @brief Tworzy nowe okno.
	 * 
	 * @param w szerokość okna
	 * @param h wysokość okna
	 * @param fs jeśli true tryb pełnoekranowy jest włączony
	 */
	Window( unsigned int w , unsigned int h , bool fs );
	/** 
	 * @brief Zamyka okno aplikacji 
	 */
	virtual ~Window();

	/** 
	 * @brief Operator sprawdzający czy okno jest poprawnie utworzone.
	 * 
	 * @return true jeśli jest, false wpp
	 */
	operator bool() {
		return !err;
	}
	
	/** 
	 * @brief Inizjalizuje okno do wyświetlania
	 */
	void init();

	/** 
	 * @brief Reaguje na zmianę wielkości okna
	 * 
	 * @param w nowa szerokość okna
	 * @param h nowa wysokość okna
	 */
	void reshape_window( unsigned int _w , unsigned int _h );

	unsigned int getW() const { return w; } /**< zwraca szerokość okna */
	unsigned int getH() const { return h; } /**< zwraca wysokość okna */
	unsigned char getErr() { return err; } /**< zwraca kod błędu okna */
private:
	bool SDL_init( unsigned int w , unsigned int h );
	bool GL_init();
	void GL_query();

	unsigned int w , h;

	unsigned char err;

	Uint32 flags;
	SDL_Surface* drawContext;
};


#endif /* __WINDOW_H__ */

