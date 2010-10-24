/**
 * @file animation.h
 * @author Jakub Kotur 
 * @version 1.0
 *
 * @section LICENSE
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of
 * the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details at
 * http://www.gnu.org/copyleft/gpl.html
 *
 * @section DESCRIPTION
 *
 * Animacje a'la c z krzyżykiem
 */

#ifndef __ANIMATION_H__

#define __ANIMATION_H__

#include <boost/signals.hpp>
#include <boost/bind.hpp>

#include "timer/timer.h"

/**
 * Szablon funktora implemetnujący liniową
 * interpolację. Działa poprawnie gdy posiada
 * następujące operatory:
 * + : (T,T)->T
 * - : (T,T)->T
 * * : (T,double)->T
 */
template<typename T>
class Lerp {
public:
	/** konstruktor */
	Lerp() {}
	/** destruktor */
	virtual ~Lerp() {}

	/** operator interpolujący
	 * @param f wartość początkowa
	 * @param t wartość końcowa
	 * @param l procent interpolacji $l\in< 0.0 , 1.0 >$
	 */
	T operator()( const T& f , const T& t , double l ) const
	{
//                if( l > 1.0 ) l = 1.0;
//                if( l < 0.0 ) l = 0.0;
		return f + (t-f)*l;
	}
};

/**
 * Klasa przeznaczona do animacji.
 * Zmienia wartość funkcji w czasie.
 * Zwykle jakiegoś settera.
 * @param T typ animowanej wartośći.
 * Musi miec zdefiniowany operator ==
 * Oraz współdziałać z funkcją interpolującą
 * @param F funkcja iterpolująca wartości
 */
template<typename T , typename F = Lerp<T> >
class Animation {
public:
	/** Typ funkcji która będzie animowana */
	typedef boost::function<void (T)> FuncSet;
	/** Typ sygnału który jest wysyłany gdy animacja
	 * dobiegnie końca */
	typedef boost::signal<void ()> SigEnd;

	/** konstruktor
	 * @param from wartość rozpoczynająca animacje
	 * @param to wartość kończąca animację
	 * @param duration czas trwania animacji w sekundach
	 * @param f funkcjia animowana
	 * @param _c falga ciągłości animacji. Jeśli true animacja
	 * nie jest przerywana po czasie duration
	 * @param _a flaga auto reverse. Jeśli true animacja 
	 * zawraca po osiągnięciu  wartości to
	 */
	Animation(const T& _f , const T& _t , double _d , const FuncSet& _s , bool _c = false , bool _a = false )
		: continuous(_c) , arev(_a) , from(_f) , to(_t) , duration(_d) , setter(_s)
	{	start(); }

	/** destruktor */
	virtual ~Animation() {}

	/** Sprawdza czy animacja jest w toku */
	bool running()
	{	return tc.good(); }

	/** Zaczyna animację */
	void start()
	{
		tc = atimer.call( boost::bind(&Animation<T>::animate,this,_1) , 0.0001 , true );
	}

	/** adwraca kierunek animowania */
	void reverse()
	{	std::swap(from,to); }

	/** kończy animację */
	void stop()
	{	tc.die(); }

	/** restartuje animate */
	void restart()
	{	stop(); start(); }

	/** ustala czy animacja ma sie wrócić */
	void auto_reverse( bool b )
	{
		arev = b;
	}
private:
	/** funkcja wywołuje funkcję interpolującą z 
	 * odpowiednimi parametrami */
	void animate( double t )
	{
		T val = interpol( from , to , t/duration );
		if( !continuous && t > duration ) {
			val = to;
			if( arev ) {
				arev = false;
				reverse();
				restart();
			} else	stop();
		} else if( arev && t > duration ) {
			reverse();
			restart();
		}
		setter( val );
	}

	F interpol;

	bool  continuous , arev;

	T from , to;
	double duration;
	FuncSet setter;

	Timer atimer;
	Timer::Caller tc;
};

#endif /* __ANIMATION_H__ */

