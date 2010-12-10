#ifndef __GFX_H__

#define __GFX_H__

#ifdef _WIN32
# include <windows.h>
#endif

#include <SDL/SDL.h>

#include <list>

#include <boost/function.hpp>
#include <boost/bind.hpp>

#include "util/config.h"

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

	void update_configuration( const Config& cfg );

	void clear() const;

	void add( Drawable* _d , int prior = 0 );
	void remove( Drawable* _d );

	void render() const;

	int width() { return mwidth; }
	int height(){ return mheight;}

	const Config& cfg() { return gfxCfg; }

	TextureManager texMgr;
	ShaderManager  shmMgr;
private:
	Config gfxCfg;

	int mwidth , mheight;

	std::list<std::pair<int,Drawable*> > to_draw;
};

} // namespace gfx

#endif /* __GFX_H__ */

