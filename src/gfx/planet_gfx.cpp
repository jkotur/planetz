#include "planet.h"

#include <fstream>
using namespace std;

#include "../sphere/sphere.h"
#include "../util/logger.h"

#include "../constants.h"

using namespace GFX;

int Planet::count = 0;

void draw_sphere( SphereModel*sm )
{
	glBegin(GL_TRIANGLES);
	for(int i = 0; i < sm->get_triangles_count(); ++i)
	{
		glTexCoord2v( sm->get_texture_point( sm->get_texture_triangle(i).p1 ) );
		glNormal3v( sm->get_normal( sm->get_triangle(i).p1 ) );
		glVertex3v( sm->get_point( sm->get_triangle(i).p1 ) );
		
		glTexCoord2v( sm->get_texture_point( sm->get_texture_triangle(i).p2 ) );
		glNormal3v( sm->get_normal( sm->get_triangle(i).p2 ) );
		glVertex3v( sm->get_point( sm->get_triangle(i).p2 ) );
		
		glTexCoord2v( sm->get_texture_point( sm->get_texture_triangle(i).p3 ) );
		glNormal3v( sm->get_normal( sm->get_triangle(i).p3 ) );
		glVertex3v( sm->get_point( sm->get_triangle(i).p3 ) );
	}
	glEnd();
}

Planet::Planet( int _d )
	: selected_b(false) , a_x(0.0) , details(1) 
	, rot_x( a_x , 90 , 3.0 , boost::bind(&Planet::set_ax,this,_1) , true )
	, tex(NULL)
{
	id = ++count;

	log_printf(INFO,"Setting planet with id %d\n",id );

	log_printf(INFO,"Makig drawing list...");
	l = glGenLists(1);
	glNewList(l,GL_COMPILE);
	draw_sphere(Sphere::get_obj(details));
	glEndList();

	log_printf(INFO,"OK\n");

	switch( rand()%4 )
	{
	case 0:
//                tex = Texture::LoadTexture( DATA("textures/planet1.png") );
		col[0]=0.7; col[1]=0.7; col[2]=0.7;
		break;
	case 1:
//                tex = Texture::LoadTexture( DATA("textures/planet2.png") );
		col[0]=0.9; col[1]=0.6; col[2]=0.3;
		break;
	case 2:
//                tex = Texture::LoadTexture( DATA("textures/planet3.png") );
		col[0]=0.9; col[1]=0.8; col[2]=0.7;
		break;
	case 3:
//                tex = Texture::LoadTexture( DATA("textures/planet2.png") );
		col[0]=0.4; col[1]=0.6; col[2]=0.8;
		break;
	}
//        tex = Texture::LoadTexture( DATA("textures/template.png") );
//        tex = Texture::LoadTexture( DATA("space_sphere.tga") );
}

Planet::~Planet()
{
}

void Planet::render( const Vector3& pos , double radius)
{
	glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

	glPushName(id);
	glPushMatrix();
	glEnable(GL_LIGHTING);
	if( tex ) {
		glEnable(GL_TEXTURE_2D);
		tex->bind();
	}
	glTranslatef( pos.x , pos.y , pos.z );
//        glRotatef(a_x,0,1,0);
	glScalef(radius,radius,radius);
	glEnable(GL_COLOR_MATERIAL);
	if( selected() )
		glColor3f( col[0]*1.2 , col[1]*1.2 , col[2]*1.2 );
	else	glColor3f( col[0] , col[1] , col[2] );
	if( l ) glCallList(l);
	else	draw_sphere(Sphere::get_obj(details));
	glDisable(GL_COLOR_MATERIAL);
	Texture::unbind();
	glPopMatrix();
	glPopName();

	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
}

