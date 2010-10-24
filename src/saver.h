#ifndef __SAVER_H__

#define __SAVER_H__

#include "ui/camera.h"
#include "planetz_manager.h"

class Saver {
public:
	Saver(  Planetz& _p ,  Camera& _c );
	virtual ~Saver();

	void save();
	void load();
	
	void save( const std::string& path );
	void load( const std::string& path );
private:
	Planetz&plz;
	Camera&cam;
};


#endif /* __SAVER_H__ */

