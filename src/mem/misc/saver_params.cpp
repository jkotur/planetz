#include "saver_params.h"

using namespace MEM::MISC;

SaverParams::SaverParams(Camera *cam)
	: planet_info( NULL )
	, cam_info( cam ) 
{
}

SaverParams::~SaverParams()
{
	if( planet_info )
	{
		delete planet_info;
	}
}
