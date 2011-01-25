#ifndef __PLANET_TRACER_H__

#define __PLANET_TRACER_H__

#include "drawable.h"

#include "util/config.h"
#include "util/timer/timer.h"

#include "mem/misc/buffer.h"
#include "mem/misc/gfx_planet_factory.h"

namespace GFX
{
/** 
 * @brief Prosta klasa odpowiedzialna za rysowanie śladów za planetami.
 */
class PlanetsTracer : public Drawable {
	public:
		/** 
		 * @brief Tworzy nowego śledzika
		 * 
		 * @param gpf pamięć graficzna karty
		 * @param num ilość śladów
		 * @param freq częstość z jaką powinny pojawiać się ślady
		 */
		PlanetsTracer( const MEM::MISC::GfxPlanetFactory& gpf
				, unsigned num = 10 , double freq = 1.0 )
			: gpf(gpf) , number(num) , oldest(0) , begin(0) , dt(freq)
			, tracing(false) , drawable(false)
		{
		}

		/** 
		 * @brief Tworzy nowgo śledzika na podstawie konfiguracji.
		 * 
		 * @param gpf pamięć graficzna karty
		 * @param cfg konfiguracja programu
		 */
		PlanetsTracer( const MEM::MISC::GfxPlanetFactory& gpf
		             , const Config& cfg )
			: gpf(gpf) , oldest(0) , begin(0)
			, tracing(false) , drawable(false)
		{
			update_configuration( cfg );
		}

		virtual ~PlanetsTracer()
		{
		}

		/** 
		 * @brief Ustawia ilość śladów
		 * 
		 * @param num nowa ilość śladów
		 */
		void setLenght( unsigned int num )
		{
			number = num;
		}

		void start(); /**< Uruchamia śledzenie */
		void stop(); /**< Zatrzymuje śledzenie */
		void clear(); /**< Czyści pamięć śledzika */

		virtual void update_configuration(); 
		void update_configuration( const Config& cfg );
		virtual void draw() const; /**< Wyświetla ślad na ekranie */
	private:
		const MEM::MISC::GfxPlanetFactory& gpf;

		void update();

		Timer::Caller tc;

		MEM::MISC::BufferGl<float3>*positions;
		unsigned number;
		unsigned oldest;
		unsigned begin;

		double dt;

		bool tracing;
		bool drawable;
};

} // GFX

#endif /* __PLANET_TRACER_H__ */

