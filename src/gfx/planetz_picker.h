#ifndef __PLANETZ_PICKER_H__

#define __PLANETZ_PICKER_H__

#include "mem/misc/gfx_planet_factory.h"
#include "gfx/shader.h"

namespace GFX
{
class PlanetzPicker {
public:
	PlanetzPicker( const MEM::MISC::GfxPlanetFactory * factory , int w , int h );
	virtual ~PlanetzPicker();

	void render( int x , int y );

	int getId();
	
private:
	GFX::ShaderManager shmMgr;
	const MEM::MISC::GfxPlanetFactory * const factory;
};

}

#endif /* __PLANETZ_PICKER_H__ */

