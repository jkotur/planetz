#pragma once
#include "planet.h"
#include "../util/logger.h"

#define PHXDT 1e-5
#define PHXG 0.666

namespace Phx
{
/** fizyczna reprezentacja modelu wszystkich planet */
class Model
{
public:
	double G; /**< stala grawitacji */
	double dt; /**< chwila */
	
	Model(){G = PHXG; dt = PHXDT;}
	
	void move();
	void add(Planet*);
	void erase(Planet*);
	void clear();
	static void set_speed(int s){ speed = s;}
private:
	std::list<Planet*> planetz; /**< lista planet */
	void move1(); /**< move per dt*/
	static int speed; /**< ilosc klatek fizyki procesowana w jednym rzucie*/
};

}
