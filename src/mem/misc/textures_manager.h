#ifndef __TEXTURES_MANAGER_H__

#define __TEXTURES_MANAGER_H__

#include <string>
#include <map>
#include <SDL/SDL_image.h>

namespace MEM
{
namespace MISC
{
	typedef std::map<unsigned, SDL_Surface*> Textures;
} // MISC
} // MEM

#endif /* __TEXTURES_MANAGER_H__ */

