#ifndef __PLANETZLAYOUT_H__

#define __PLANETZLAYOUT_H__

#include <boost/signal.hpp>
#include <boost/regex.hpp>

#include <string>

#include "util/config.h"

#include "layout.h"

//typedef boost::signal<void (Planet*p)> SigSetPlanet;
typedef boost::signal<void (std::string)> SigSetString;
typedef boost::signal<void (double)> SigSetDouble;
typedef boost::signal<void ()> SigVoid;

/** 
 * @brief Klasa odpowiedzialna za stworzenie konkretnego GUI i 
 * obsługę zdarzeń z nim związanych.
 */
class PlanetzLayout : public Layout {
public:
	/** 
	 * @brief Tworzy guziki i okienka na podstawie konfiguracji.
	 * 
	 * @param cfg konfiguracja początkowa programu
	 */
	PlanetzLayout( Config& cfg );
	virtual ~PlanetzLayout();
	
	//void add_selected_planet( Planet*p );

	//SigSetPlanet on_planet_delete;
	//SigSetPlanet on_planet_add;
	/** @brief Sygnał emitowany gdy zmienia się prędkość kamery */
	SigSetDouble on_cam_speed_changed;
	/** @brief Sygnał emitowany gdy symulacja jest pauzowana */
	SigVoid      on_pause_click;
	/** @brief Sygnał emitowany gdy symulacja jest resetowana */
	SigVoid      on_reset_click;
	/** @brief Sygnał emitowany gdy układ jest zapisywany */
	SigSetString on_save;
	/** @brief Sygnał emitowany gdy układ jest wczytywany */
	SigSetString on_load;
	/** @brief Sygnał emitowany gdy zmienia się konfiguracja programu */
	boost::signal<void ( const Config&)> on_config_changed;

	/** 
	 * @brief Funkcja ustawiająca nową ilość wyświetlonych klatek.
	 * 
	 * @param fps ilość klatek wyświetlonych przez sekundę.
	 */
	void update_fps( int fps );
private:
	Config& config;

	void updateOptions( Config& cfg );
	void setOptions( const Config& cfg );

	bool clear_win( const CEGUI::EventArgs& e );
	bool add_planet( const CEGUI::EventArgs& e );
	bool del_planet( const CEGUI::EventArgs& e );
	bool show_load_win( const CEGUI::EventArgs& e );
	bool show_save_win( const CEGUI::EventArgs& e );
	bool show_opt_win( const CEGUI::EventArgs& e );
	bool hide_opt_win( const CEGUI::EventArgs& e );
	bool apply_options( const CEGUI::EventArgs& e );
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

