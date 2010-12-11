
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <cmath>
#include <cstring>
#include <sstream>

#include "planetzlayout.h"

#include "util/logger.h"
#include "util/vector.h"
#include "util/timer/timer.h"

#include "constants.h"

#define BUFSIZE 128

#define GETWIN(x)(WindowManager::getSingleton().getWindow(x))
#define GETWINCAST(t,x) static_cast<t >(GETWIN(x))

#define ERRORWIN(x) \
	do { \
		GETWIN("stError")->setText(x); \
		GETWIN("ErrorWin")->setVisible(true); \
	} while( 0 ) 

#define SETEVENT(x,y,z) GETWIN(x)->subscribeEvent((y),Event::Subscriber((z),this));

#ifdef _WIN32
# define snprintf sprintf_s
#endif

using namespace CEGUI;

using namespace boost::filesystem;

using std::pow;

class MyListboxItem : public ListboxTextItem  {
	public:
		MyListboxItem( const CEGUI::String& text )
			: ListboxTextItem(text)
		{
			setSelectionBrushImage ("QuadraticLook", "White");
			setSelectionColours (CEGUI::colour (0.0, 0.118, 1.0, 0.412));
		}
};

double mass_pow( double x )
{
	return pow( 2 , x );
}

const boost::regex PlanetzLayout::save_file("[\\w ]+.sav");
const boost::regex PlanetzLayout::file_cont("[\\w ]+");
const std::string PlanetzLayout::qsave_name = "qsave.sav";

PlanetzLayout::PlanetzLayout()
	: Layout("planetz.layout") //, sel_planet(NULL)
{
	WindowManager::getSingleton().getWindow("btnAdd")
		->subscribeEvent(PushButton::EventClicked
				,Event::Subscriber(&PlanetzLayout::add_planet,this));

	WindowManager::getSingleton().getWindow("btnPause")
		->subscribeEvent(PushButton::EventClicked
				,Event::Subscriber(&PlanetzLayout::pause,this));

	WindowManager::getSingleton().getWindow("btnSave")
		->subscribeEvent(PushButton::EventClicked
				,Event::Subscriber(&PlanetzLayout::save,this));

	WindowManager::getSingleton().getWindow("btnShowLoad")
		->subscribeEvent(PushButton::EventClicked
				,Event::Subscriber(&PlanetzLayout::show_load_win,this));

	SETEVENT("btnShowSave",PushButton::EventClicked,&PlanetzLayout::show_save_win);

	SETEVENT("btnShowOpt" ,PushButton::EventClicked,&PlanetzLayout::show_opt_win);
	SETEVENT("btnOptNope" ,PushButton::EventClicked,&PlanetzLayout::hide_opt_win);
	SETEVENT("btnOptOk" ,PushButton::EventClicked,&PlanetzLayout::apply_options);

	WindowManager::getSingleton().getWindow("btnLoad")
		->subscribeEvent(PushButton::EventClicked
				,Event::Subscriber(&PlanetzLayout::load,this));

	WindowManager::getSingleton().getWindow("btnReset")
		->subscribeEvent(PushButton::EventClicked
				,Event::Subscriber(&PlanetzLayout::reset_anim,this));

	WindowManager::getSingleton().getWindow("btnClear")
		->subscribeEvent(PushButton::EventClicked
				,Event::Subscriber(&PlanetzLayout::clear_win,this));

	WindowManager::getSingleton().getWindow("btnDel")
		->subscribeEvent(PushButton::EventClicked
				,Event::Subscriber(&PlanetzLayout::del_planet,this));

	SETEVENT("ErrorWin",FrameWindow::EventCloseClicked,&PlanetzLayout::close_win);
	SETEVENT("SavesListWin",FrameWindow::EventCloseClicked,&PlanetzLayout::close_win);
	SETEVENT("winSave",FrameWindow::EventCloseClicked,&PlanetzLayout::close_win);

	WindowManager::getSingleton().getWindow("slCamSpeed")
		->subscribeEvent(Scrollbar::EventScrollPositionChanged
				,Event::Subscriber(&PlanetzLayout::set_cam_speed,this));

	WindowManager::getSingleton().getWindow("slAnimSpeed")
		->subscribeEvent(Scrollbar::EventScrollPositionChanged
				,Event::Subscriber(&PlanetzLayout::set_anim_speed,this));

	WindowManager::getSingleton().getWindow("slMass")
		->subscribeEvent(Scrollbar::EventScrollPositionChanged
				,Event::Subscriber(&PlanetzLayout::set_mass_val,this));

	WindowManager::getSingleton().getWindow("slRadius")
		->subscribeEvent(Scrollbar::EventScrollPositionChanged
				,Event::Subscriber(&PlanetzLayout::set_radius_val,this));

	SETEVENT("btnQSave",PushButton::EventClicked,&PlanetzLayout::qsave);
	SETEVENT("btnQLoad",PushButton::EventClicked,&PlanetzLayout::qload);

//        timer.call( boost::bind(&PlanetzLayout::update_show_window,this) , 0.1 , true );
}

PlanetzLayout::~PlanetzLayout()
{
}

Config PlanetzLayout::getOptions()
{
	Config cfg;
	cfg.set("textures",GETWINCAST(Checkbox*,"cbTextures")->isSelected());
	cfg.set("lightsplanes",GETWINCAST(Checkbox*,"cbLights")->isSelected());
	cfg.set("lighting",GETWINCAST(Checkbox*,"cbLighting")->isSelected());
	return cfg;
}

void PlanetzLayout::setOptions( const Config& cfg )
{
	GETWINCAST(Checkbox*,"cbTextures")->setSelected(cfg.get<bool>("textures"));
	GETWINCAST(Checkbox*,"cbLights")->setSelected(cfg.get<bool>("lightsplanes"));
	GETWINCAST(Checkbox*,"cbLighting")->setSelected(cfg.get<bool>("lighting"));
}

/*void PlanetzLayout::add_selected_planet( Planet*p )
{
	sel_planet = p;
	update_show_window();
}*/

void PlanetzLayout::update_show_window()
{
	return;
	/*if( ! sel_planet ) {
		WindowManager::getSingleton().getWindow("ShowWin")->setVisible(false);
		return;
	}

	WindowManager::getSingleton().getWindow("ShowWin")->setVisible(true);

	char buff[BUFSIZE];

	::Vector3 pos = sel_planet->get_phx()->get_pos();
	snprintf(buff,BUFSIZE,"(%4.2f,%4.2f,%4.2f)",pos.x,pos.y,pos.z);
	WindowManager::getSingleton().getWindow("lbShowPos")->setText(buff);

	snprintf(buff,BUFSIZE,"%4.2f",sel_planet->get_phx()->get_radius() );
	WindowManager::getSingleton().getWindow("lbShowRadius")->setText(buff);

	::Vector3 vel = sel_planet->get_phx()->get_velocity();
	snprintf(buff,BUFSIZE,"(%4.2f,%4.2f,%4.2f)",vel.x,vel.y,vel.z);
	WindowManager::getSingleton().getWindow("lbShowSpeedVector")->setText(buff);

	snprintf(buff,BUFSIZE,"%4.2f",vel.length());
	WindowManager::getSingleton().getWindow("lbShowSpeedScalar")->setText(buff);

	snprintf(buff,BUFSIZE,"%4.2f",sel_planet->get_phx()->get_mass() );
	WindowManager::getSingleton().getWindow("lbShowMass")->setText(buff);

	snprintf(buff,BUFSIZE,"Aktualna planeta (%d)",sel_planet->get_id());
	GETWIN("ShowWin")->setText(buff);*/
}

void PlanetzLayout::update_fps( int fps )
{
	std::stringstream ss;
	ss << "FPS: " << fps;
	WindowManager::getSingleton().getWindow("stFps")->setText(ss.str());
}

bool PlanetzLayout::del_planet( const CEGUI::EventArgs& e )
{
	log_printf(DBG,"[GUI] Removin planet!\n");
	//on_planet_delete(sel_planet);
	WindowManager::getSingleton().getWindow("stFps")->setVisible(false);
	//sel_planet = NULL;
	return true;
}

bool PlanetzLayout::add_planet( const CEGUI::EventArgs& e )
{
	//        Listbox*lb = static_cast<Listbox*>(WindowManager::getSingleton().getWindow("lbox1"));
	//        lb->addItem( new ListboxTextItem("Item") );
	//        return true;
	::Vector3 pos;
	pos.x=static_cast<Spinner*>(WindowManager::getSingleton().getWindow("spPosX"))
		->getCurrentValue();
	pos.y=static_cast<Spinner*>(WindowManager::getSingleton().getWindow("spPosY"))
		->getCurrentValue();
	pos.z=static_cast<Spinner*>(WindowManager::getSingleton().getWindow("spPosZ"))
		->getCurrentValue();

	::Vector3 vel;
	vel.x=static_cast<Spinner*>(WindowManager::getSingleton().getWindow("spSpeedX"))
		->getCurrentValue();
	vel.y=static_cast<Spinner*>(WindowManager::getSingleton().getWindow("spSpeedY"))
		->getCurrentValue();
	vel.z=static_cast<Spinner*>(WindowManager::getSingleton().getWindow("spSpeedZ"))
		->getCurrentValue();

	double mass;
	mass=static_cast<Scrollbar*>(WindowManager::getSingleton().getWindow("slMass"))
		->getScrollPosition();
	mass = mass_pow(mass);

	double radius;
	radius=static_cast<Scrollbar*>(WindowManager::getSingleton().getWindow("slRadius"))
		->getScrollPosition();
	radius++; // minimalny promien to 1

	log_printf(DBG,"[GUI] Adding planet at (%f,%f,%f) with speed (%f,%f,%f), mass %f and radius %f\n"
			,pos.x,pos.y,pos.z
			,vel.x,vel.y,vel.z
			,mass,radius );

	//GFX::Planet*gp = new GFX::Planet( );
	//Phx::Planet*pp = new Phx::Planet( pos , vel , mass , radius );
	//on_planet_add( new Planet(gp,pp) );

	return true;
}

bool PlanetzLayout::set_cam_speed( const CEGUI::EventArgs& e )
{
	on_cam_speed_changed( 
			static_cast<Scrollbar*>(WindowManager::getSingleton().getWindow("slCamSpeed"))
			->getScrollPosition() );
	return true;
}

bool PlanetzLayout::clear_win( const CEGUI::EventArgs& e )
{
	static_cast<Spinner*>(WindowManager::getSingleton().getWindow("spPosX"))
		->setCurrentValue( 0 );
	static_cast<Spinner*>(WindowManager::getSingleton().getWindow("spPosY"))
		->setCurrentValue( 0 );
	static_cast<Spinner*>(WindowManager::getSingleton().getWindow("spPosZ"))
		->setCurrentValue( 0 );

	static_cast<Spinner*>(WindowManager::getSingleton().getWindow("spSpeedX"))
		->setCurrentValue(0);
	static_cast<Spinner*>(WindowManager::getSingleton().getWindow("spSpeedY"))
		->setCurrentValue(0);
	static_cast<Spinner*>(WindowManager::getSingleton().getWindow("spSpeedZ"))
		->setCurrentValue( 0 );

	static_cast<Scrollbar*>(WindowManager::getSingleton().getWindow("slMass"))
		->setScrollPosition( 0 );

	static_cast<Scrollbar*>(WindowManager::getSingleton().getWindow("slRadius"))
		->setScrollPosition( 0 );

	return true;
}

bool PlanetzLayout::set_mass_val( const CEGUI::EventArgs& e )
{
	double mass=static_cast<Scrollbar*>(WindowManager::getSingleton().getWindow("slMass"))
		->getScrollPosition();
	mass = mass_pow(mass);
	char buff[BUFSIZE];
	snprintf(buff,BUFSIZE,"%4.2f",mass);
	WindowManager::getSingleton().getWindow("stMassCount")
		->setText(buff);
	return true;
}

bool PlanetzLayout::set_radius_val( const CEGUI::EventArgs& e )
{
	double radius=static_cast<Scrollbar*>(WindowManager::getSingleton().getWindow("slRadius"))
		->getScrollPosition();
	radius++;
	char buff[BUFSIZE];
	snprintf(buff,BUFSIZE,"%4.2f",radius);
	WindowManager::getSingleton().getWindow("stRadiusCount")
		->setText(buff);
	return true;
}

bool PlanetzLayout::pause( const CEGUI::EventArgs& e )
{
	CEGUI::Window* btn = GETWIN("btnPause");
	if( btn->getText() == "Start" )
		btn->setText("Pauza");
	else	btn->setText("Start");
	on_pause_click();
	return true;
}

bool PlanetzLayout::show_load_win( const CEGUI::EventArgs& e )
{
	Listbox*lb = static_cast<Listbox*>(WindowManager::getSingleton().getWindow("lstSaves"));
	lb->resetList();
	if( !exists( SAVES("") ) )  {
		ERRORWIN("Brak folderu z zapisanymi układami");
		return true;
	}

	
	directory_iterator end_itr; // default construction yields past-the-end
	for ( directory_iterator itr( SAVES("") ); itr != end_itr; ++itr )
	{
		if( !is_directory( itr->status() )
		 && boost::regex_match(itr->filename().c_str(),save_file) )
			lb->addItem(new MyListboxItem(itr->filename()));
	}

	GETWIN("SavesListWin")->setVisible(true);

	return true;
}

bool PlanetzLayout::load( const CEGUI::EventArgs& e )
{
	GETWIN("SavesListWin")->setVisible(false);

	ListboxItem*lbi = GETWINCAST(Listbox*,"lstSaves")->getFirstSelectedItem();
	
	if( lbi ) {
		GETWIN("btnPause")->setText("Start");
		on_load(SAVES(lbi->getText().c_str()));
	}

	return true;
}

bool PlanetzLayout::qload( const CEGUI::EventArgs& e )
{
	on_load( SAVES(qsave_name) );
	return true;
}

bool PlanetzLayout::show_save_win( const CEGUI::EventArgs& e )
{
	GETWIN("winSave")->setVisible(true);
	return true;
}

bool PlanetzLayout::show_opt_win( const CEGUI::EventArgs& e )
{
	setOptions( config );
	GETWIN("winOpt")->setVisible(!GETWIN("winOpt")->isVisible());
	return true;
}

bool PlanetzLayout::hide_opt_win( const CEGUI::EventArgs& e )
{
	GETWIN("winOpt")->setVisible(false);
	return true;
}

bool PlanetzLayout::apply_options( const CEGUI::EventArgs& e )
{
//        GETWIN("winOpt")->setVisible(false);
	config = getOptions();
	on_config_changed( config );
	return true;
}

bool PlanetzLayout::save( const CEGUI::EventArgs& e )
{
	std::string str = GETWIN("ebSave")->getText().c_str();
	GETWIN("winSave")->setVisible(false);

	if( !regex_match(str,save_file) ) {
		if( regex_match(str,file_cont) )
			str+=".sav";
		else { 
			GETWIN("ebSave")->setText("");
			ERRORWIN("Niepoprawna nazwa pliku");
			return true;
		}
	}

	log_printf(DBG,"Saving %s\n", str.c_str() );
	on_save( SAVES(str) );
	return true;
}

bool PlanetzLayout::qsave( const CEGUI::EventArgs& e )
{
	on_save( SAVES(qsave_name) );
	return true;
}

bool PlanetzLayout::reset_anim( const CEGUI::EventArgs& e )
{
	on_reset_click();
	return true;
}

bool PlanetzLayout::set_anim_speed( const CEGUI::EventArgs& e )
{
	//Phx::Model::set_speed(
	//		static_cast<Scrollbar*>(WindowManager::getSingleton().getWindow("slAnimSpeed"))
	//		->getScrollPosition() + 100 );
	return true;
}

bool PlanetzLayout::close_win( const CEGUI::EventArgs& e )
{
	static_cast<const WindowEventArgs&>(e).window->setVisible(false);
	return true;
}

