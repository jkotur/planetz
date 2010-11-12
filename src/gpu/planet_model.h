
#ifndef __PLANET_MODEL_H__

#define __PLANET_MODEL_H__

#include <GL/glew.h>

namespace GPU
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
} // GPU

#endif /* __PLANET_MODEL_H__ */

