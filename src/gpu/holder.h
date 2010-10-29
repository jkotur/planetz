#ifndef HOLDER_H
#define HOLDER_H

#include "buffer.h"

namespace GPU
{
	struct Holder
	{
		Holder( unsigned num  = 0 )
			: planet_model      (num) ,
			  planet_pos        (num) ,
			  planet_radius     (num) ,
                          planet_count      (num) ,
                                            
                                            
                          planet_mass       (num) ,
                          planet_velocity   (num) ,
                                            
                          pointsCloud_points(num) ,
                          pointsCloud_size  (num)
		{
		}

		void resize( const size_t num )
		{
			planet_model      .resize(num);
			planet_pos        .resize(num);
			planet_radius     .resize(num);
			planet_count      .resize(num);
				    
				    
			planet_mass       .resize(num);
			planet_velocity   .resize(num);
				    
			pointsCloud_points.resize(num);
			pointsCloud_size  .resize(num);
		}

		// Planet
		//   * GFX
		BufferGl<uint8_t>  planet_model;

		//   * COMMON
		BufferGl<float3>   planet_pos;
		BufferGl<float>    planet_radius;
		BufferGl<uint32_t> planet_count;

		//   * PHX
		BufferCu<float>    planet_mass;
		BufferCu<float3>   planet_velocity;

		BufferGl<float3>   pointsCloud_points;
		BufferGl<uint32_t> pointsCloud_size;
	};
}

#endif // HOLDER_H
