
#ifndef __PLANET_MODEL_H__

#define __PLANET_MODEL_H__

#include <GL/glew.h>

namespace MEM
{
namespace MISC
{
	class PlanetzModel {
	public:
		PlanetzModel()
			: vertices(NULL)
			, texCoord(NULL)
			, len(0)
			, part_len(0)
		{}

		GLuint* vertices;
		GLuint* texCoord;
		GLsizei len     ;
		GLsizei part_len;
		GLsizei parts   ;
	};
} // MEM::MISC
} // MEM

#endif /* __PLANET_MODEL_H__ */

