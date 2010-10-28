#ifndef __ARROW_H__

#define __ARROW_H__

#include <GL/glew.h>

#include "../util/vector.h"

namespace GFX {

class Arrow {
public:
	Arrow ();
	virtual ~Arrow();

	void render( const Vector3& pos , const Vector3& v );
private:
	void draw_tube( const Vector3& v);

	GLint list;
};

} // namespace GFX


#endif /* __ARROW_H__ */

