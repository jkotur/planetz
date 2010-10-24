#ifndef __PLANET_H__

#define __PLANET_H__

#include "./phx/planet.h"
#include "./gfx/planet.h"

#include "./gfx/arrow.h"

/**
 * klasa laczaca wygląd planety z jego fizyką
 */
class Planet {
public:
	Planet( Gfx::Planet*gp , Phx::Planet*lp );
	virtual ~Planet();

	void render();

	/**
	 * Obiekt gry jest skasowany gdy
	 * skasowany zostanie on sam, lub
	 * jego obiekt logiki.
	 */
	bool deleted()
	{
		return mdel || phx_obj->deleted();
	}

	void del() 
	{	mdel = true; }

	void trace()
	{	mtrace = true; }
	void untrace()
	{	mtrace = false; }
	bool tracing()
	{	return mtrace; }

	void select()
	{	if( gfx_obj ) gfx_obj->select(); }
	void deselect()
	{	if( gfx_obj ) gfx_obj->deselect(); }
	int get_id()
	{	return gfx_obj?gfx_obj->get_id():-1; }
	Phx::Planet*get_phx()
	{	return phx_obj; }
	Gfx::Planet*get_gfx()
	{	return gfx_obj; }
private:
	bool mdel; /**< flaga skasowania */
	bool mtrace;

	Gfx::Arrow arr;

	Gfx::Planet * gfx_obj; /**< graficzna reprezentacja planety */
	Phx::Planet * phx_obj; /**< fizyczna reprezentacja planety */
};


#endif /* __PLANET_H__ */

