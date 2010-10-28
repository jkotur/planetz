#include "arrow.h"

#include <cmath>

#include "../constants.h"

using namespace GFX;
using std::sin;
using std::cos;

Arrow::Arrow()
{
}

Arrow::~Arrow()
{
}

//void Arrow::draw_tube()
//{
//        double r = 1.0;
//        double y = 1.0;
//        int faceCount = 20;

//        glBegin(GL_QUAD_STRIP);
//        for (int i = 0; i <= faceCount; i++)
//        {
//                double t = (double)i/faceCount;
//                double x = r*cos(t*PI2);
//                double z = r*sin(t*PI2);

//                glVertex3f( x , y , z );
//                glVertex3f( x , -y , z );

//        }
//        glEnd();

//}

void Arrow::draw_tube( const Vector3& v )
{
	double r = 1.0;
	int faceCount = 20;

	glBegin(GL_QUAD_STRIP);
	for (int i = 0; i <= faceCount; i++)
	{
		double t = (double)i/faceCount;
		double x = r*cos(t*PI2);
		double z = r*sin(t*PI2);

		Vector3 v1 = Vector3( x, 0 , z ).normalize();
		Vector3 v2 = Vector3( x, 1 , z ).normalize();
		v2.cross( v );
		v2 *= v.length();
		glVertex3v( v1 );
		glVertex3v( v2 );

	}
	glEnd();

}


void Arrow::render( const Vector3& pos , const Vector3& v )
{
	glPushMatrix();

	glDisable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);

	glColor3f( 0.7 , 0.5 , 0.2 );

	Vector3 vs = v + Vector3(0,0,1);

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

//        draw_tube( v );

	glPopMatrix();
}

