#ifndef __DRAWABLE_H__

#define __DRAWABLE_H__

#include <cstdlib>

namespace GFX
{
	class Gfx;

	class Drawable {
	public:
		Drawable () : gfx(NULL) {}
		virtual ~Drawable() {}
		
		virtual void draw() const =0;
		virtual void prepare() {}

		virtual void resize(
				unsigned int width ,
				unsigned int height) {}

		virtual void setGfx( Gfx* _g )
		{	gfx = _g; prepare(); }
	protected:
		Gfx* gfx;
	};
}

#endif /* __DRAWABLE_H__ */

