#ifndef __PHX_PLANET_H__

#define __PHX_PLANET_H__

#include "../util/vector.h"

#include <list>

namespace Phx {

/** fizyczna reprezentacja planety */
class Planet {
public:
	/** konstruktor */
	Planet( const Vector3& pos /**< pozycja startowa planety */
	   , const Vector3& speed /**< początkowy wektor predkosci planety */
	   , double m /**< masa planety */
	   , double r /**< promien*/);
	virtual ~Planet();

	bool deleted()
	{	return mdel; }

	void del()
	{	mdel = true; }

	Vector3 get_pos()
	{	return pos; }

	void set_pos( const Vector3& v )
	{
		pos = v;
	}
	double get_mass()
	{
		return m;
	}
	double get_radius()
	{
		return radius;
	}
	Vector3 get_velocity()
	{
		return speed;
	}
	
	friend class Model;

private:
	bool mdel; /**< flaga skasowania */
	Vector3 pos; /**< akutalna pozycja */
	Vector3 speed; /**< aktualna predkosc */
	Vector3 force; /**< aktualna sila */
	double m; /**< masa planety */
	double radius;
};

}


#endif /* __PHX_PLANET_H__ */

