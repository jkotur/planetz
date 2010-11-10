#ifndef __ARROW_H__

#define __ARROW_H__

#include <GL/glew.h>

#include "drawable.h"
#include "../util/vector.h"

namespace GFX {

class Arrow : public Drawable {
public:
	Arrow( );
	Arrow( const Vector3& v ) : v(v) , color(v) {}
	virtual ~Arrow();

	void draw() const
	{
		render( Vector3() , v );
	}

	void render( const Vector3& pos , const Vector3& v ) const;
private:
	const Vector3 v;
	const Vector3 color;

	GLint list;
};

} // namespace GFX


#endif /* __ARROW_H__ */

