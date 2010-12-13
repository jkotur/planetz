#ifndef __WINDOW_H__

#define __WINDOW_H__

#include <SDL/SDL.h>

#include <vector>

class Window {
public:
	Window( const std::vector<int>& res , bool fs );
	Window( unsigned int , unsigned int , bool fs );
	virtual ~Window();

	operator bool() {
		return !err;
	}
	
	void init();
	bool SDL_init( unsigned int w , unsigned int h );
	bool GL_init();
	void GL_query();

	void reshape_window( unsigned int _w , unsigned int _h );

	unsigned int getW() const { return w; }
	unsigned int getH() const { return h; }
	unsigned char getErr() { return err; }
private:
	unsigned int w , h;

	unsigned char err;

	Uint32 flags;
	SDL_Surface* drawContext;
};


#endif /* __WINDOW_H__ */

