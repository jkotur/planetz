
#ifndef __UI_H__

#define __UI_H__

#include <boost/signals.hpp>
#include <boost/bind.hpp>

#include "input/driver.h"

#include "gfx/drawable.h"

#include "./gui/gui.h"
#include "./camera.h"

class UI : public GFX::Drawable {
	struct breaker {                                                                       
		typedef void result_type;                                        

		template<typename InputIterator>                                
		result_type operator()(InputIterator first, InputIterator last) const   
		{
			while( first != last && *first == false ) ++first;
		}
	};
public:
	UI ();
	virtual ~UI();

	bool init();

	virtual void draw() const;

	void signal();

	int event_handle();
	
	boost::signal<void (int,int,int)> sigKeyUp;
	boost::signal<void (SDLKey,Uint16,Uint8)> sigKeyDown;
	boost::signal<void (int,int)> sigMouseMotion;
	boost::signal<void (int,int,int)> sigMouseButtonUp;
	boost::signal<bool (int,int,int) , breaker > sigMouseButtonDown;
	boost::signal<void (int,int)> sigVideoResize;

	Gui gui;
private:
	CLocationDriver*joy;
};

#endif /* __UI_H__ */

