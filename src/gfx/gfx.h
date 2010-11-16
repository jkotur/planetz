#ifndef __GFX_H__

#define __GFX_H__

#ifdef _WIN32
# include <windows.h>
#endif

#include <SDL/SDL.h>

#include <list>

#include <boost/function.hpp>
#include <boost/bind.hpp>

#include "drawable.h"

#include "shader.h"
#include "texture.h"

/**
 * Przestrzeń nazw dla obiektów odpowiedzialnych za grafikę
 */
namespace GFX {

/**
 * Główna klasa odpowiedzialna za wyświetlanie
 */
class Gfx {
public:
	virtual ~Gfx();

	bool GL_view_init();

	void GL_viewport( int  w , int h);

	bool window_init(int width,int height);

	void reshape_window(int w, int h);

	void clear() const;

	void add( Drawable* _d );
	void remove( Drawable* _d );

	void render() const;

	int width() { return mwidth; }
	int height(){ return mheight;}

	TextureManager texMgr;
	ShaderManager  shmMgr;
private:
	int mwidth , mheight;
	SDL_Surface* drawContext;
	Uint32 flags;

	std::list<Drawable*> to_draw;

//        Camera*c;
};

} // namespace gfx

#endif /* __GFX_H__ */

