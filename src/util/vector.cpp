#ifdef _WIN32
# include <windows.h>
#endif

#include <GL/gl.h>

#include "vector.h"

Vector3 operator-( const Vector3& a , const Vector3& b )
{
	return Vector3( a.x-b.x , a.y-b.y , a.z-b.z );
}

Vector3 operator+( const Vector3& a , const Vector3& b )
{
	return Vector3( a.x+b.x , a.y+b.y , a.z+b.z );
}

Vector3 operator/( const Vector3& a , const int b)
{
	return Vector3( a.x/b , a.y/b , a.z/b);
}

bool operator==( const Vector3& a , const Vector3& b )
{
	return vcompare(a, b);
}

bool operator!=( const Vector3& a , const Vector3& b )
{
	return !vcompare(a, b);
}

bool vcompare ( const Vector3& a , const Vector3& b )
{
	return (abs(a.x - b.x) < VECTOREPSILON && abs(a.y - b.y) < VECTOREPSILON && abs(a.z - b.z) < VECTOREPSILON);
}

void glVertex3v( const Vector3& v )
{
	glVertex3f( v.x , v.y , v.z );
}

void glNormal3v( const Vector3& v )
{
	glNormal3f( v.x , v.y , v.z );
}

