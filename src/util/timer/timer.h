/**
 * @file timer.h
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
 */

#ifndef __TIMER_H__

#define __TIMER_H__

#include <stdlib.h>
#ifndef _WIN32
# include <sys/time.h>
#else
# include <windows.h>
#endif
#include <list>
#include <vector>
#ifndef _NON_BOOST
# include <boost/function.hpp>
# include <queue>
#endif

/**
 * Lista stanów zegara
 */
enum States {
	RUNNING =0,
	STOPED,
	PAUSED
};

#ifndef _NON_BOOST
template <class T>
struct cmp_func : std::binary_function <T,T,bool> {
	bool operator() (const T& x, const T& y) const
		// odwrocona kolejnosc
		{	return x->time>y->time; }
};
#endif

/**
 * Klasa implementujaca licznik
 *
 * @note probably works fine on windows also, but never tested
 */
class Timer {
#ifndef _NON_BOOST
	struct TimeFunc;
#endif
public:
#ifndef _NON_BOOST
	typedef boost::function<void ( double )> Function;
	typedef std::priority_queue<TimeFunc*,std::vector<TimeFunc*>,cmp_func<TimeFunc*> > ToCallQueue;
#endif
	/** 
	 * @brief Obiekt dający kontrolę nad zaplanowanymi w timerze zdarzeniami
	 */
	class Caller {
	public:
		Caller( States*_s =NULL) : state(_s) {}
		Caller( const Caller& c ) : state(c.state) {}
		/** 
		 * @brief Sprawdza czy obiekt jest poprawnym obiektem zdarzenia.
		 * 
		 * @return true jeśli jest dobry, false wpp
		 */
		bool good()
		{	return state!=NULL; }
		/** 
		 * @brief Sprawdza czy zdarzenie uktualnie działa
		 * 
		 * @return true jeśli tak, false wpp
		 */
		bool running()
		{	return state && *state==RUNNING; }
		/** 
		 * @brief Kasuje zdarzenie z kolejki wywołań. Po wywołaniu tej funkcji
		 * obiekt staje się bezużyteczny.
		 */
		void die()
		{	if( state ) {
			    *state = STOPED;
			    state = NULL;
			}
		}
		/** 
		 * @brief Blockuje zdarzenie. Nie będzie ono wykonywane do czasu odblokowania.
		 */
		void block()
		{	( state && (*state = PAUSED) );}
		/** 
		 * @brief Odblokowuje zdarzenie.
		 */
		void unblock()
		{	( state && (*state = RUNNING));}
	private:
		States*state;
	};

	Timer( bool _start = true , bool _sigable = true );
	~Timer();

	void start();
	void stop();

	void pause();
	void unpause();

	void signal();

	double get(); /**< get time in seconds */
	double get_s(); /**< get time in seconds */
	double get_ms(); /**< get time in mili seconds */
	double get_mms(); /**< get time in micro seconds */

	double get_dt(); /**< get time in seconds */
	double get_dt_s(); /**< get time in seconds */
	double get_dt_ms(); /**< get time in mili seconds */
	double get_dt_mms(); /**< get time in micro seconds */

#ifndef _NON_BOOST
	/**
	 * Funkcja dodaję żądanie wywołania funkcji. Czas którym operuje
	 * podany powinień być w sekundach
	 */
	Caller call( const Function& function /**< funkcja która ma być wywołana */
		 , double interval /**< interwał po jakim ma być wywołana */
		 , bool repeat = false /**< czy wywoływanie ma być powtarzane*/ );
	/**
	 * Funkcja dodaję żądanie wywołania funkcji. Czas którym operuje
	 * podany powinień być w sekundach
	 */
	Caller call_at( const Function& function /**< funkcja która ma być wywołana */
		    , double time /**< Czaw bezwzględny w którym ma być wywołana */ );
#endif
private:
	States state;
	double start_time_mms;
	double end_time_mms;
	double old_dt_mms;
	double new_dt_mms;
	double paused_time_mms;
	bool sigable;
#ifdef _WIN32
	LARGE_INTEGER freq;
	LARGE_INTEGER start_count;
	LARGE_INTEGER end_count;
#else
	struct timeval start_count;
	struct timeval end_count;
#endif

#ifndef _NON_BOOST 
	struct TimeFunc {
		TimeFunc( double t , double i , const  Function& f )
			: start_time(t) , time(t) , interval(i) , func(f) , state(RUNNING) {}
		double start_time;
		double time;
		double interval;
		Function func;
		States state;
	};

	ToCallQueue to_call;

	friend struct cmp_func<TimeFunc*>;
#endif

	double mms_to_s( double a )
	{
		return a*0.000001;
	}

	double mms_to_ms( double a )
	{
		return a*0.001;
	}

	static std::list<Timer*> timers;
public:
	static void signal_all();
};

extern Timer timer;

#endif /* __TIMER_H__ */

