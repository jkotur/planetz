#include "model.h"

using namespace Phx;

int Model::speed = 100;

void Model::add(Planet* pl)
{
	planetz.push_back(pl);
}

void Model::erase(Planet* pl)
{
	planetz.remove(pl);
}

void Model::clear()
{
	planetz.clear();
}

void Model::move()
{
	dt = (double)speed/5000.0 * PHXDT;
	for(int i = 0; i < Model::speed; ++i)
		move1();
}

void Model::move1()
{
	for(std::list<Planet*>::iterator pl = planetz.begin(); pl != planetz.end(); ++pl)
	{
		if((*pl)->deleted())
			continue;
		Vector3 actForce(0, 0, 0);
		for(std::list<Planet*>::iterator nextpl = planetz.begin(); nextpl != planetz.end(); ++nextpl)
		{
			if(pl == nextpl || (*nextpl)->deleted())
				continue;
			Vector3 diff = (*nextpl)->pos - (*pl)->pos;
			double dist2 = diff.length();
			
			if(std::max((*nextpl)->radius, (*pl)->radius) > dist2) //kolizja
			{
				if((*nextpl)->radius > (*pl)->radius)
				{
					Planet* tmp = *pl;
					*pl = *nextpl;
					*nextpl = tmp;
				}
				double a1 = (*pl)->radius / ((*pl)->radius + (*nextpl)->radius), a2 = (*nextpl)->radius / ((*pl)->radius + (*nextpl)->radius);
				double b1 = (*pl)->m / ((*pl)->m + (*nextpl)->m), b2 = (*nextpl)->m / ((*pl)->m + (*nextpl)->m);
				//log_printf(DBG, "(%f, %f, %f) + (%f, %f, %f) => ", (*pl)->pos.x, (*pl)->pos.y, (*pl)->pos.z, (*nextpl)->pos.x, (*nextpl)->pos.y, (*nextpl)->pos.z);
				(*pl)->pos = (*pl)->pos * a1 + (*nextpl)->pos * a2;//tak naprawdę wygląda lepiej
				//log_printf(DBG, "(%f, %f, %f)\n", (*pl)->pos.x, (*pl)->pos.y, (*pl)->pos.z);
				(*pl)->speed = (*pl)->speed * b1 + (*nextpl)->speed * b2;
				(*pl)->m += (*nextpl)->m;
				(*pl)->radius = pow( pow((*pl)->radius, 3.0) + pow((*nextpl)->radius, 3.0) , 1.0 / 3.0);
				(*nextpl)->del();
				
				log_printf(DBG, "  [PHX] Removin planet!\n");
			}
			else
			{
				dist2 *= dist2;
				diff.normalize();
				double diffLen = G * (*pl)->m * (*nextpl)->m / dist2;
				diff *= diffLen;
			
				actForce += diff;
			}
		}
		(*pl)->force = actForce;
	}
	for(std::list<Planet*>::iterator pl = planetz.begin(); pl != planetz.end(); ++pl)
	{
		Vector3 newspeed = (*pl)->speed + ((*pl)->force / (*pl)->m) * dt;
		(*pl)->pos += (( newspeed + (*pl)->speed) * dt ) / 2;
		(*pl)->speed = newspeed;
	}
}
