#include "planetz_renderer.h"

using namespace GFX;

PlanetzRenderer::PlanetzRenderer( const GPU::GfxPlanetFactory * factory )
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

void PlanetzRenderer::setModels( GPU::PlanetzModel mod )
{
	grend.setModels( mod );
}

void PlanetzRenderer::prepare()
{
}

void PlanetzRenderer::draw() const
{
//        crend.draw();
//        grend.draw();
	drend.draw();
}

