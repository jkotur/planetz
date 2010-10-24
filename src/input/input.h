#ifndef __INPUT_H__

#define __INPUT_H__

#include <boost/signals.hpp>
#include "driver.h"

struct breaker {                                                                       
	typedef void result_type;                                        

	template<typename InputIterator>                                
	result_type operator()(InputIterator first, InputIterator last) const   
	{
		while( first != last && *first == false ) ++first;
	}
};

class CInput {
public:
	CInput();
	virtual ~CInput();

	CLocationDriver*joy;
};

int event_handle();

extern CInput input;
extern boost::signal<void (int,int,int)> SigKeyUp;
extern boost::signal<void (SDLKey,Uint16,Uint8)> SigKeyDown;
extern boost::signal<void (int,int)> SigMouseMotion;
extern boost::signal<void (int,int,int)> SigMouseButtonUp;
extern boost::signal<bool (int,int,int) , breaker > SigMouseButtonDown;
extern boost::signal<void (int,int)> SigVideoResize;

#endif /* __INPUT_H__ */

