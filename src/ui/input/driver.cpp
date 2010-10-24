#include "driver.h"
#include "gfx/gfx.h"
#include "util/vector.h"
#include "util/logger.h"

bool CNullLD::Init()
{
	return true;
}

Vector3 CNullLD::GetPosition()
{
	return Vector3(0, 0, 0);
}

bool CJoystickLD::Init()
{
	if (SDL_NumJoysticks())
	{
		mJoystick = SDL_JoystickOpen(0);
		return true;
	}
	else
	{
		return false;
	}
}

// WTF: too many magic numbers:O
Vector3 CJoystickLD::GetPosition()
{
	Sint16 i = SDL_JoystickGetAxis(mJoystick,  0);
	float f0 = (float)i / 32768.0f;
	i = SDL_JoystickGetAxis(mJoystick,  1);
	float f1 = (float)i / 32768.0f;
	i = SDL_JoystickGetAxis(mJoystick,  3);
	float f2 = (float)i / 32768.0f;

	double m = 150;

	Vector3 ret(0,0,0);

	/*if (SDL_JoystickGetButton(mJoystick, 4))
	{
		Vector3 start = Vector3(300, 50, 0);
		ret = Vector3(-10.005, 75, -0.05) - start;
	}*/

	ret.z += -f0 * m ;//* 5;
	ret.y += -f1 * m;// * 5;
	ret.x += f2 * m ;//* 5;

	return ret;
}

bool CMouseLD::Init()
{
	return true;
}

Vector3 CMouseLD::GetPosition()
{
	int x , y ;
	int z = SDL_GetMouseState(&x, &y)&SDL_BUTTON(SDL_BUTTON_LEFT)?50:0;
	return Vector3( x - gfx.width()/2 , gfx.height()/2 - y , z );
}

