#ifndef __PLANETZLAYOUT_H__

#define __PLANETZLAYOUT_H__

#include <boost/signal.hpp>
#include <boost/regex.hpp>

#include <string>

#include "layout.h"

//typedef boost::signal<void (Planet*p)> SigSetPlanet;
typedef boost::signal<void (std::string)> SigSetString;
typedef boost::signal<void (double)> SigSetDouble;
typedef boost::signal<void ()> SigVoid;

class PlanetzLayout : public Layout {
public:
	PlanetzLayout ();
	virtual ~PlanetzLayout();
	
	//void add_selected_planet( Planet*p );

	//SigSetPlanet on_planet_delete;
	//SigSetPlanet on_planet_add;
	SigSetDouble on_cam_speed_changed;
	SigVoid on_pause_click;
	SigVoid on_reset_click;
	SigSetString on_save;
	SigSetString on_load;

	void update_fps( int fps );
private:
	bool clear_win( const CEGUI::EventArgs& e );
	bool add_planet( const CEGUI::EventArgs& e );
	bool del_planet( const CEGUI::EventArgs& e );
	bool show_load_win( const CEGUI::EventArgs& e );
	bool show_save_win( const CEGUI::EventArgs& e );
	bool set_cam_speed( const CEGUI::EventArgs& e );
	bool pause( const CEGUI::EventArgs& e );
	bool save( const CEGUI::EventArgs& e );
	bool load( const CEGUI::EventArgs& e );
	bool reset_anim( const CEGUI::EventArgs& e );
	bool set_anim_speed( const CEGUI::EventArgs& e );
	bool close_win( const CEGUI::EventArgs& e );
	bool qsave( const CEGUI::EventArgs& e );
	bool qload( const CEGUI::EventArgs& e );

	bool set_mass_val( const CEGUI::EventArgs& e );
	bool set_radius_val( const CEGUI::EventArgs& e );


	void update_show_window();

	//Planet*sel_planet;

	static const boost::regex save_file;
	static const boost::regex file_cont;
	static const std::string qsave_name;
};


#endif /* __PLANETZLAYOUT_H__ */

