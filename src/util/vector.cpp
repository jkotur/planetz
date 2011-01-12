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

Vector3 mul4f( const Vector3& v , const float * m )
{
	Vector3 o = Vector3(
		v.x*m[ 0]+v.y*m[ 4]+v.z*m[ 8]+m[12] ,
		v.x*m[ 1]+v.y*m[ 5]+v.z*m[ 9]+m[13] ,
		v.x*m[ 2]+v.y*m[ 6]+v.z*m[10]+m[14] );
	float w=v.x*m[ 3]+v.y*m[ 7]+v.z*m[11]+m[15] ;
	o /= w;
	return o;
}

Vector3 mul4f( const float * m , const Vector3& v )
{
	Vector3 o = Vector3(
		v.x*m[ 0]+v.y*m[ 1]+v.z*m[ 2]+m[ 3] ,
		v.x*m[ 4]+v.y*m[ 5]+v.z*m[ 6]+m[ 7] ,
		v.x*m[ 8]+v.y*m[ 9]+v.z*m[10]+m[11] );
	float w=v.x*m[12]+v.y*m[13]+v.z*m[14]+m[15] ;
	o /= w;
	return o;
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

void glTexCoord2v( const Vector3& v  )
{
	glTexCoord2d( v.x , v.y );
}

