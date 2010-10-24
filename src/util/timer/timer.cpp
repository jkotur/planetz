#include <stdlib.h>

#include "timer.h"

std::list<Timer*> Timer::timers;

Timer timer;

Timer::Timer( bool _start , bool _sigable )
	: state(STOPED) , start_time_mms(0) , end_time_mms(0) , paused_time_mms(0) , sigable(_sigable)
{
	if( _sigable ) timers.push_back(this);

#ifdef _WIN32
	QueryPerformanceFrequency(&freq);
	start_count.QuadPart = 0;
	end_count.QuadPart = 0;
#else
	start_count.tv_sec = start_count.tv_usec = 0;
	  end_count.tv_sec =   end_count.tv_usec = 0;
#endif

	if( _start ) start();
}

Timer::~Timer()
{
#ifndef _NON_BOOST 
	while( !to_call.empty() ) {
		delete to_call.top();
		to_call.pop();
	}
#endif
	if( sigable ) timers.remove(this);
}

void Timer::start()
{
	state = RUNNING;
#ifdef WIN32
	QueryPerformanceCounter(&start_count);
#else
	gettimeofday(&start_count,NULL);
#endif
	old_dt_mms = new_dt_mms = get_mms();
}

void Timer::stop()
{
	get_mms(); // sets end_time_mms to current
	state = STOPED;
#ifdef WIN32
	QueryPerformanceCounter(&end_count);
#else
	gettimeofday(&end_count,NULL);
#endif
}

void Timer::pause()
{
	if( state != RUNNING ) return;
	stop();
	state = PAUSED;
}

void Timer::unpause()
{
	if( state != PAUSED ) return;
	double end_time_mms_tmp = end_time_mms;
	state = RUNNING;
	get_mms();
	double paused_dt = end_time_mms - end_time_mms_tmp;
	paused_time_mms += paused_dt;
	old_dt_mms += paused_dt;
	new_dt_mms += paused_dt;
}

void Timer::signal()
{
	if( state != RUNNING ) return;

	// update dt
	old_dt_mms = new_dt_mms;
	new_dt_mms = get_mms();

#ifndef _NON_BOOST 
	if( to_call.empty() ) return;

	// call functions
	double curr = get_s() - mms_to_s(paused_time_mms);
	while( to_call.size() && to_call.top()->time <= curr )
	{
		TimeFunc*tf = to_call.top();
		to_call.pop();
		if( tf->state == STOPED ) continue;
		if( tf->state == RUNNING) tf->func( curr - tf->start_time );
		if( tf->interval ) {
			// or maybe tf->time = curr + tf->interval ???
			tf->time += tf->interval;
			to_call.push(tf);
		} else	delete tf;
	}
#endif
}

void Timer::signal_all()
{
	for( std::list<Timer*>::iterator i = timers.begin() ; i != timers.end() ; ++i )
		(*i)->signal();
}

double Timer::get()
{
	return get_s();
}

double Timer::get_s()
{
	return mms_to_s(get_mms());
}

double Timer::get_ms()
{
	return mms_to_ms(get_mms());
}

double Timer::get_mms()
{
#ifdef WIN32
	if(state==RUNNING) QueryPerformanceCounter(&end_count);
	start_time_mms = start_count.QuadPart * (1000000.0 / freq.QuadPart);
	end_time_mms   =   end_count.QuadPart * (1000000.0 / freq.QuadPart);
#else
	if(state==RUNNING) gettimeofday(&end_count, NULL);
	start_time_mms = (start_count.tv_sec *1000000.0) + start_count.tv_usec;
	end_time_mms   = (  end_count.tv_sec *1000000.0) +   end_count.tv_usec;
#endif

	return end_time_mms - start_time_mms;
}

double Timer::get_dt()
{
	return get_dt_s();
}

double Timer::get_dt_s()
{
	return get_dt_mms() * 0.000001;
}

double Timer::get_dt_ms()
{
	return  get_dt_mms() * 0.001;
}

double Timer::get_dt_mms()
{
	return new_dt_mms - old_dt_mms;
}

#ifndef _NON_BOOST 

Timer::Caller Timer::call( const Function& func , double interval , bool repeat )
{
	// push function without paused time
	TimeFunc*tf = new TimeFunc(	get_s()-mms_to_s(paused_time_mms)+interval,
					repeat?interval:0,
					func);
	to_call.push(tf);
	return Caller(&(tf->state));
}

Timer::Caller Timer::call_at( const Function& func , double time )
{
	TimeFunc*tf = new TimeFunc(time-mms_to_s(paused_time_mms),0,func);
	to_call.push(tf);
	return Caller(&tf->state);
}

#endif
