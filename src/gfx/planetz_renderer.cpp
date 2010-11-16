#include "planetz_renderer.h"

using namespace GFX;

PlanetzRenderer::PlanetzRenderer( const MEM::MISC::GfxPlanetFactory * factory )
	: grend  ( factory )
	, crend  ( factory )
	, drend  ( factory )
	, factory( factory )
{
}

PlanetzRenderer::~PlanetzRenderer()
{
	log_printf(DBG,"[DEL] Deleting PlanetzRenderer\n");
}

void PlanetzRenderer::setModels( MEM::MISC::PlanetzModel mod )
{
	grend.setModels( mod );
}

void PlanetzRenderer::prepare()
{
	drend.prepare();
}

void PlanetzRenderer::setGfx( Gfx * _g )
{
	Drawable::setGfx( _g );

	drend.setGfx( _g );
}

void PlanetzRenderer::draw() const
{
//        crend.draw();
//        grend.draw();
	drend.draw();
}

