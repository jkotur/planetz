#ifndef __PLANETZ_MANAGER_H__

#define __PLANETZ_MANAGER_H__

#include <list>	
#include <boost/signal.hpp>

#include "./planet.h"
#include "phx/model.h"
#include "camera.h"

typedef boost::signal<void (Planet*)> SigPlanet;

class Planetz {
public:
	typedef std::list<Planet*>::iterator iterator;

	Planetz();
//        Planetz( Camera* c );
	virtual ~Planetz();

	iterator begin()
	{	return planetz.begin(); }

	iterator end()
	{	return planetz.end(); }

	void add( Planet* );
	void erase( Planet* );

	void clear();

	void render();
	void update();

//        void lookat();

	void select(int);

	SigPlanet on_planet_select;
private:
//        const Camera*main_cam;
//        Camera*curr_cam;
//        Camera planet_cam;
	Phx::Model *phx_model;
	std::list<Planet*> planetz;
};

#endif /* __PLANETZ_MANAGER_H__ */

