#include <boost/foreach.hpp>

#include "./planetz_manager.h"
#include "./util/logger.h"

Planetz::Planetz()
//Planetz::Planetz( Camera* cam )
//        : main_cam(cam) , curr_cam(cam)
//          , planet_cam(Vector3(0,0,0),Vector3(0,0,0),Vector3(0,0,0) )
{
	phx_model = new Phx::Model();
}

Planetz::~Planetz()
{
}

void Planetz::add( Planet*p )
{
	phx_model->add(p->get_phx());
	planetz.push_back(p);
}

void Planetz::clear()
{
	select(-1);

	phx_model->clear();
	BOOST_FOREACH( Planet*p , planetz )
		delete p;
	planetz.clear();
}

void Planetz::erase( Planet*p )
{
	phx_model->erase(p->get_phx());
	planetz.remove(p);
	delete p;
}

void Planetz::draw() const
{
	BOOST_FOREACH( Planet*p , planetz )
		p->render();
}

void Planetz::update()
{
	phx_model->move();

//        curr_cam = const_cast<Camera*>(main_cam);

	for( std::list<Planet*>::iterator i = planetz.begin() ; i != planetz.end() ; ++i )
	{
//                if( (*i)->tracing() ) {

//                        Phx::Planet*o = (*i)->get_phx();
//                        curr_cam = &planet_cam;

//                        curr_cam->set_perspective(
//                                        o->get_pos() , 
//                                        o->get_velocity() + o->get_pos(),
//                                        main_cam->get_up() );
//                }

		if( (*i)->deleted() ) {
			delete *i;
			phx_model->erase((*i)->get_phx());
			i = planetz.erase(i);
			select(-1);
		}

		if( i == planetz.end() ) break;
	}
}

void Planetz::select( int id )
{
	Planet*sel = NULL;
	BOOST_FOREACH( Planet*p , planetz )
	{
		if( p->get_id() == id ) {
			p->select();
			sel = p;
		} else	p->deselect();
	}
	on_planet_select(sel);
}

//void Planetz::lookat()
//{
//        curr_cam->gl_lookat();
//}

