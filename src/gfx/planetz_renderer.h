#ifndef __PLANETZ_RENDERER_H__

#define __PLANETZ_RENDERER_H__

namespace GFX {

	class PlanetzRenderer : public Drawable {
	public:
		PlanetzRenderer( const GPU::GfxPlanetFactory * factory );
		virtual ~PlanetzRenderer();
		
		virtual void draw() const;
	private:
		const GPU::GfxPlanetFactory * factory;
	};

}

#endif /* __PLANETZ_RENDERER_H__ */

