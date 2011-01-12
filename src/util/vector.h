#ifndef __VECTOR_H__

#define __VECTOR_H__

#include <cmath>
#define VECTOREPSILON 1e-12

using std::sin;
using std::cos;
using std::sqrt;
using std::abs;

/** 
 * @brief Implementacja trójwymiarowego wektora matematycznego
 */
class Vector3 {
public:
	double x,y,z;
	Vector3(double x = 0, double y = 0, double z = 0): x(x), y(y), z(z){}
	Vector3(const Vector3 &copy)
	{
		x=copy.x;
		y=copy.y;
		z=copy.z;
	}
	/** 
	 * @brief Normalizuje wektor
	 */
	const Vector3& normalize() {

		double len = sqrt(x*x + y*y + z*z);
		if (len > 0.00001) {
			x /= len;
			y /= len;
			z /= len;
		}
		return *this;
	}
	/** 
	 * @return długość wektora
	 */
	double length() const
	{	return sqrt(x*x + y*y + z*z); }
	const Vector3& operator += (const Vector3 &v2)
	{
		x+=v2.x;
		y+=v2.y;
		z+=v2.z;
		return (*this);
	}
	const Vector3& operator -= (const Vector3 &v2)
	{
		x-=v2.x;
		y-=v2.y;
		z-=v2.z;
		return (*this);
	}
	const Vector3& operator /=(double s)
	{
		x/=s;
		y/=s;
		z/=s;
		return (*this);
	}
	const Vector3& operator *=(double s)
	{
		x*=s;
		y*=s;
		z*=s;
		return (*this);
	}
	Vector3 operator-()
	{
		return Vector3(-x,-y,-z);
	}
	Vector3 operator * (double s)
	{
		Vector3 out( x*s , y*s , z*s );
		return out;
	}
	/** 
	 * @return zwraca iloczyn skalarny dwóch wektorów
	 */
	double dot(const Vector3 &v2)
	{
		return x*v2.x + y*v2.y + z*v2.z;
	}
	/** 
	 * @return iloczyn skalarany po osiach x i z
	 */
	double dotXZ(const Vector3 &v2)
	{
		return x*v2.x + z*v2.z;
	}
	/** 
	 * @brief obraca wektor wokół osi OX
	 * 
	 * @param angle kąt obrotu wyrażony w radianach
	 */
	void rotateX(double angle)
	{
		double newy = cos(angle) * y - sin(angle) * z;
		double newz = sin(angle) * y + cos(angle) * z;
		y=newy;
		z=newz;
	}
	/** 
	 * @brief obraca wektor wokół osi OY
	 * 
	 * @param angle kąt obrotu wyrażony w radianach
	 */
	void rotateY(double angle)
	{
		double newx = cos(angle) * x - sin(angle) * z;
		double newz = sin(angle) * x + cos(angle) * z;
		x = newx;
		z = newz;
	}
	/** 
	 * @brief obraca wektor wokół osi OZ
	 * 
	 * @param angle kąt obrotu wyrażony w radianach
	 */
	void rotateZ(double angle)
	{
		double newx = cos(angle) * x - sin(angle) * y;
		double newy = sin(angle) * x + cos(angle) * y;
		x=newx;
		y=newy;
	}

	/** 
	 * @brief Obraca wektor wokół zadanej osi
	 * 
	 * @param u oś obrotu
	 * @param angle kąt obrotu wyrażony w radianach
	 */
	void rotate( const Vector3& u , double angle )
	{
		double c = cos(angle);
		double s = sin(angle);

		double R[3][3] =
		{
			{ u.x*u.x+(1-u.x*u.x)*c , u.x*u.y*(1-c)-u.z*s , u.x*u.z*(1-c)+u.y*s } ,
			{ u.x*u.y*(1-c)+u.z*s , u.y*u.y+(1-u.y*u.y)*c , u.y*u.z*(1-c)-u.x*s } ,
			{ u.x*u.z*(1-c)-u.y*s , u.y*u.z*(1-c)+u.x*s , u.z*u.z+(1-u.z*u.z)*c }
		};

		double tx = x*R[0][0] + y*R[1][0] + z*R[2][0];
		double ty = x*R[0][1] + y*R[1][1] + z*R[2][1];
		double tz = x*R[0][2] + y*R[1][2] + z*R[2][2];

		x = tx; y=ty;z=tz;
	}

	/* double XZangle(Vector3 v2)
	{
		Vector3 p1 = *this, p2 = v2;
		//p1.y=p2.y=0;
		//double d=  p1.distance(p2);
		double w = p1.x-p2.x;
		double ww = p1.z-p2.z;
		double r =atan2(ww,w);
			// asin(w/d);
//                if (r<0) 
//                        r+=2*M_PI;
		return r;
	}
	*/

	/** 
	 * @return odległość między wektorami
	 */
	double distance(const Vector3 &v2)
	{
		double p = x-v2.x , q = y-v2.y, r=z-v2.z;
		return sqrt(p*p + q*q + r*r);
	}
	/** 
	 * @return odległość między wektorami w osiach OX i OZ
	 */
	double distanceXZ(const Vector3 &v2)
	{
		double p = x-v2.x , r=z-v2.z;
		return sqrt(p*p + r*r);
	}
	/** 
	 * @brief Oblicza iloczyn wektorowy z drugim wektorem
	 * 
	 * @param v2 drugi wektor
	 */
	void cross( const Vector3&v2 )
	{
		double tx = y*v2.z - z*v2.y;
		double ty = z*v2.x - x*v2.z;
		double tz = x*v2.y - y*v2.x;
		x = tx; y = ty; z = tz;
	}

//        Vector3 operator-( const Vector3& a )
//        {
//                return Vector3( a.x-x , a.y-y , a.z-z );
//        }

//        Vector3 operator+( const Vector3& a )
//        {
//                return Vector3( a.x+x , a.y+y , a.z+z );
//        }

};

Vector3 operator+( const Vector3& a , const Vector3& b );
Vector3 operator-( const Vector3& a , const Vector3& b );
Vector3 operator/( const Vector3& a , const int b);
Vector3 mul4f( const float * m , const Vector3& v );
Vector3 mul4f( const Vector3& v , const float * m );
bool operator==( const Vector3& a , const Vector3& b );
bool operator!=( const Vector3& a , const Vector3& b );
bool vcompare(const Vector3& a , const Vector3& b );
void glVertex3v( const Vector3& v );
void glNormal3v( const Vector3& v );
void glTexCoord2v( const Vector3& v  );

//void glTranslatev( const Vector3& v )
//{
//        glTranslatef( v.x , v.y , v.z );
//}

#endif /* __VECTOR_H__ */

