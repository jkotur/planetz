#include "planetz_renderer.h"

using namespace GFX;

PlanetzRenderer::PlanetzRenderer( const MEM::MISC::GfxPlanetFactory * factory )
	: drend  ( factory )
	, factory( factory )
{
}

PlanetzRenderer::~PlanetzRenderer()
{
	log_printf(DBG,"[DEL] Deleting PlanetzRenderer\n");
}

void PlanetzRenderer::setMaterials( GLuint matTex )
{
	drend.setMaterials( matTex );
}

void PlanetzRenderer::prepare()
{
}

void PlanetzRenderer::resize( unsigned int width , unsigned int height )
{
	drend.resize(width,height);
}

void PlanetzRenderer::on_camera_angle_changed( float*m )
{
	drend.on_camera_angle_changed( m );
}

void PlanetzRenderer::setGfx( Gfx * _g )
{
	Drawable::setGfx( _g );

	drend.setGfx( _g );
}

void PlanetzRenderer::draw() const
{
	drend.draw();
}

