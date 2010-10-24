#ifndef __DRIVER_H__

#define __DRIVER_H__

#include "SDL/SDL.h"
#include "../util/vector.h"

class CLocationDriver
{
public:
	virtual ~CLocationDriver() {}
	virtual bool Calibrate() { return true; }
	virtual bool Init() = 0;
	virtual Vector3 GetPosition() = 0;
	virtual void GetPosition(Vector3 &ret)
	{	ret = GetPosition(); }
};

class CNullLD : public CLocationDriver
{
public:
	virtual ~CNullLD() {}
	virtual bool Init();
	virtual Vector3 GetPosition();
};


class CJoystickLD : public CLocationDriver
{
	SDL_Joystick *mJoystick;
public:
	virtual ~CJoystickLD() {}
	virtual bool Init();
	virtual Vector3 GetPosition();
};

class CMouseLD : public CLocationDriver {
public:
	virtual ~CMouseLD() {}
	virtual bool Init();
	virtual Vector3 GetPosition();
};

#endif /* __DRIVER_H__ */

