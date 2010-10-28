#ifndef __DRAWABLE_H__

#define __DRAWABLE_H__

namespace GFX
{
	class Gfx;

	class Drawable {
	public:
		Drawable () : gfx(NULL) {}
		virtual ~Drawable() {}
		
		virtual void draw() const =0;

		void setGfx( const Gfx* _g )
		{	gfx = _g; }
	protected:
		const Gfx* gfx;
	};
}

#endif /* __DRAWABLE_H__ */

