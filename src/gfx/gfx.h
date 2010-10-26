#ifndef __GFX_H__

#define __GFX_H__

#ifdef _WIN32
# include <windows.h>
#endif

#include <SDL/SDL.h>

#include <boost/function.hpp>
#include <boost/bind.hpp>

/**
 * Przestrzeń nazw dla obiektów odpowiedzialnych za grafikę
 */
namespace Gfx {

/**
 * Główna klasa odpowiedzialna za wyświetlanie
 */
class CGfx {
public:
	virtual ~CGfx();

	bool SDL_init(int w,int h);

	bool GL_init();

	bool GL_view_init();

	void GL_viewport( int  w , int h);

	void reshape_window(int w, int h);

	void clear();

	int width() { return mwidth; }
	int height(){ return mheight;}
private:
	int mwidth , mheight;
	SDL_Surface* drawContext;
	Uint32 flags;

//        Camera*c;
};

} // namespace gfx

#endif /* __GFX_H__ */

