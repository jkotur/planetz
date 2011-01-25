#include "planetz_setter.h"

#include "constants.h"
#include "gfx/gfx.h"
#include "util/logger.h"

UI::PlanetzSetter::PlanetzSetter()
	: Z(0)
{
	mode = MODE_NONE;

	quad = gluNewQuadric();
}

UI::PlanetzSetter::~PlanetzSetter()
{
	gluDeleteQuadric(quad);
}

void UI::PlanetzSetter::draw() const
{
	if( mode == MODE_NONE ) return;

	glPushMatrix();
	glDisable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
	glTranslatef(position.x,position.y,position.z);
	drawRadius();
	drawVel();
	glPopMatrix();
}

void UI::PlanetzSetter::drawRadius() const 
{
	glColor3f( .40f , .45f , .5f );
	glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	gluSphere(quad,radius,32,32);
	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
}

void UI::PlanetzSetter::drawVel() const 
{

	// FIXME: this is buggy couse arrowhead may be missing
	Vector3 vs = velocity + Vector3(0.666,0.1337,0);

	vs.cross( velocity );
	vs.normalize();
	vs *= velocity.length()*0.15;

	glColor3f( .70f , .85f , .95f );

	glBegin(GL_LINES); 
	glVertex3f( 0,0,0 );
	glVertex3v( velocity );

	glVertex3v( velocity );
	glVertex3v( (velocity + vs)*0.8 );

	glVertex3v( velocity );
	glVertex3v( (velocity - vs)*0.8 );
	glEnd();
}

void UI::PlanetzSetter::signal()
{
}

void UI::PlanetzSetter::on_mouse_motion( int x , int y )
{
	if( mode == MODE_NONE ) return;

	switch( mode )
	{
		case MODE_POS:
		case MODE_POS_ONLY:
			position_mv = screen_camera( x , y , Z );
			position = camera_world( position_mv );
			break;
		case MODE_VEL:
		case MODE_VEL_ONLY:
			{
			Vector3 v;
			GLfloat mv[16];
			glGetFloatv(GL_MODELVIEW_MATRIX,mv);
			Vector3 pos = mul4f(position,mv);
			GLfloat p [16];
			glGetFloatv(GL_PROJECTION_MATRIX,p);
			v = Vector3(0,0,pos.z);
			v = mul4f(v,p);
			velocity = screen_camera( x , y , v.z );
			velocity = camera_world( velocity ) - position;
			}
			break;
		case MODE_RADIUS:
		case MODE_RADIUS_ONLY:
			{
			Vector3 v;
			GLfloat mv[16];
			glGetFloatv(GL_MODELVIEW_MATRIX,mv);
			Vector3 pos = mul4f(position,mv);
			GLfloat p [16];
			glGetFloatv(GL_PROJECTION_MATRIX,p);
			v = Vector3(0,0,pos.z);
			v = mul4f(v,p);
			v = screen_camera( x , y , v.z);
			radius = (v-pos).length();
			}
			break;
		default:
			break;
	}

	MEM::MISC::PlanetParams pp;
	pp.pos = make_float3(position.x,position.y,position.z);
	pp.vel = make_float3(velocity.x,velocity.y,velocity.z);
	pp.radius = radius;

	on_planet_changed( pp );
}

void UI::PlanetzSetter::update( const MEM::MISC::PlanetParams& pp )
{
	velocity = Vector3( pp.vel.x , pp.vel.y , pp.vel.z );
	position = Vector3( pp.pos.x , pp.pos.y , pp.pos.z );
	radius   = pp.radius;
}

bool UI::PlanetzSetter::on_button_down( int b , int x , int y )
{
	if( mode == MODE_NONE || mode == MODE_DRAW ) return false;
	if( b != SET_BUTTON ) return false;
	mode = (enum MODE)((int)mode + 1);
	if( (int)mode >= (int)MODE_COUNT ) {
		log_printf(DBG,"position: %f %f %f\n",position.x,position.y,position.z);
		log_printf(DBG,"radius: %f\n",radius);
		mode = MODE_NONE;
	}
	return true;
}

void UI::PlanetzSetter::change( const MEM::MISC::PlanetParams& pp )
{
	mode = MODE_POS;

	update(pp);

	GLfloat  p [16];
	glGetFloatv(GL_PROJECTION_MATRIX,p);
	Vector3 v(0,0,-5*radius);
	v = mul4f(v,p);
	Z = v.z;
}

Vector3 UI::PlanetzSetter::screen_camera( int x , int y , float z )
{
	Vector3 out;

	out.x =(       x / (float)gfx->width () - .5f)*2.0f;
	out.y =(0.5f - y / (float)gfx->height()      )*2.0f;
	out.z = z;

	GLfloat  p [16];
	GLfloat  pi[16];

	glGetFloatv(GL_PROJECTION_MATRIX,p);

	inverse(  pi ,  p );

	out = mul4f( out ,  pi );

	return out;
}

Vector3 UI::PlanetzSetter::camera_world( const Vector3& in )
{
	GLfloat mv [16];
	GLfloat mvi[16];

	glGetFloatv(GL_MODELVIEW_MATRIX,mv);

	inverse( mvi , mv );

	return mul4f( in , mvi );
}

/*
 * Compute the inverse of a 4x4 matrix.
 *
 * From an algorithm by V. Strassen, 1969, _Numerishe Mathematik_, vol. 13,
 * pp. 354-356.
 * 60 multiplies, 24 additions, 10 subtractions, 8 negations, 2 divisions,
 * 48 assignments, _0_ branches
 *
 * This implementation by Scott McCaskill
 */
typedef GLfloat Mat2[2][2];

enum {
    M00 = 0, M01 = 4, M02 = 8, M03 = 12,
    M10 = 1, M11 = 5, M12 = 9, M13 = 13,
    M20 = 2, M21 = 6, M22 = 10,M23 = 14,
    M30 = 3, M31 = 7, M32 = 11,M33 = 15
};

void UI::PlanetzSetter::inverse( GLfloat dst[16] , const GLfloat src[16] )
{
   Mat2 r1, r2, r3, r4, r5, r6, r7;
   const GLfloat * A = src;
   GLfloat *       C = dst;
   GLfloat one_over_det;

   /*
    * A is the 4x4 source matrix (to be inverted).
    * C is the 4x4 destination matrix
    * a11 is the 2x2 matrix in the upper left quadrant of A
    * a12 is the 2x2 matrix in the upper right quadrant of A
    * a21 is the 2x2 matrix in the lower left quadrant of A
    * a22 is the 2x2 matrix in the lower right quadrant of A
    * similarly, cXX are the 2x2 quadrants of the destination matrix
    */

   /* R1 = inverse( a11 ) */
   one_over_det = 1.0f / ( ( A[M00] * A[M11] ) - ( A[M10] * A[M01] ) );
   r1[0][0] = one_over_det * A[M11];
   r1[0][1] = one_over_det * -A[M01];
   r1[1][0] = one_over_det * -A[M10];
   r1[1][1] = one_over_det * A[M00];

   /* R2 = a21 x R1 */
   r2[0][0] = A[M20] * r1[0][0] + A[M21] * r1[1][0];
   r2[0][1] = A[M20] * r1[0][1] + A[M21] * r1[1][1];
   r2[1][0] = A[M30] * r1[0][0] + A[M31] * r1[1][0];
   r2[1][1] = A[M30] * r1[0][1] + A[M31] * r1[1][1];

   /* R3 = R1 x a12 */
   r3[0][0] = r1[0][0] * A[M02] + r1[0][1] * A[M12];
   r3[0][1] = r1[0][0] * A[M03] + r1[0][1] * A[M13];
   r3[1][0] = r1[1][0] * A[M02] + r1[1][1] * A[M12];
   r3[1][1] = r1[1][0] * A[M03] + r1[1][1] * A[M13];

   /* R4 = a21 x R3 */
   r4[0][0] = A[M20] * r3[0][0] + A[M21] * r3[1][0];
   r4[0][1] = A[M20] * r3[0][1] + A[M21] * r3[1][1];
   r4[1][0] = A[M30] * r3[0][0] + A[M31] * r3[1][0];
   r4[1][1] = A[M30] * r3[0][1] + A[M31] * r3[1][1];

   /* R5 = R4 - a22 */
   r5[0][0] = r4[0][0] - A[M22];
   r5[0][1] = r4[0][1] - A[M23];
   r5[1][0] = r4[1][0] - A[M32];
   r5[1][1] = r4[1][1] - A[M33];

   /* R6 = inverse( R5 ) */
   one_over_det = 1.0f / ( ( r5[0][0] * r5[1][1] ) - ( r5[1][0] * r5[0][1] ) );
   r6[0][0] = one_over_det * r5[1][1];
   r6[0][1] = one_over_det * -r5[0][1];
   r6[1][0] = one_over_det * -r5[1][0];
   r6[1][1] = one_over_det * r5[0][0];

   /* c12 = R3 x R6 */
   C[M02] = r3[0][0] * r6[0][0] + r3[0][1] * r6[1][0];
   C[M03] = r3[0][0] * r6[0][1] + r3[0][1] * r6[1][1];
   C[M12] = r3[1][0] * r6[0][0] + r3[1][1] * r6[1][0];
   C[M13] = r3[1][0] * r6[0][1] + r3[1][1] * r6[1][1];

   /* c21 = R6 x R2 */
   C[M20] = r6[0][0] * r2[0][0] + r6[0][1] * r2[1][0];
   C[M21] = r6[0][0] * r2[0][1] + r6[0][1] * r2[1][1];
   C[M30] = r6[1][0] * r2[0][0] + r6[1][1] * r2[1][0];
   C[M31] = r6[1][0] * r2[0][1] + r6[1][1] * r2[1][1];

   /* R7 = R3 x c21 */
   r7[0][0] = r3[0][0] * C[M20] + r3[0][1] * C[M30];
   r7[0][1] = r3[0][0] * C[M21] + r3[0][1] * C[M31];
   r7[1][0] = r3[1][0] * C[M20] + r3[1][1] * C[M30];
   r7[1][1] = r3[1][0] * C[M21] + r3[1][1] * C[M31];

   /* c11 = R1 - R7 */
   C[M00] = r1[0][0] - r7[0][0];
   C[M01] = r1[0][1] - r7[0][1];
   C[M10] = r1[1][0] - r7[1][0];
   C[M11] = r1[1][1] - r7[1][1];

   /* c22 = -R6 */
   C[M22] = -r6[0][0];
   C[M23] = -r6[0][1];
   C[M32] = -r6[1][0];
   C[M33] = -r6[1][1];
}

