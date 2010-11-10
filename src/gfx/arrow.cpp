#include "arrow.h"

#include <cmath>

#include "../constants.h"

using namespace GFX;
using std::sin;
using std::cos;

Arrow::Arrow()
	: color( 0.7 , 0.5 , 0.2 )
{
}

Arrow::~Arrow()
{
}

void Arrow::render( const Vector3& pos , const Vector3& v ) const
{
	glPushMatrix();

	glDisable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);

	glColor3f( color.x , color.y , color.z );

	// FIXME: this is buggy couse arrowhead may be missing
	Vector3 vs = v + Vector3(0.666,0.1337,0);

	vs.cross( v );
	vs.normalize();
	vs *= v.length()*0.15;

	glTranslatef( pos.x , pos.y , pos.z );
	glBegin(GL_LINES);
	  glVertex3f( 0,0,0 );
	  glVertex3v( v );

	  glVertex3v( v );
	  glVertex3v( (v + vs)*0.8 );

	  glVertex3v( v );
	  glVertex3v( (v - vs)*0.8 );
	glEnd();

	glPopMatrix();
}

