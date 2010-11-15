#include "copy_renderer.h"

#include "sphere/sphere.h"
#include "util/vector.h"

using namespace GFX;

void draw_sphere_2( SphereModel*sm )
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

CopyRender::CopyRender( const MEM::MISC::GfxPlanetFactory * factory ) 
	: factory( factory )
{
	sphereListId = glGenLists(1);
	glNewList(sphereListId,GL_COMPILE);
	draw_sphere_2(Sphere::get_obj(0));
	glEndList();
}

CopyRender::~CopyRender()
{
}

void CopyRender::draw() const
{
	if( factory->getPositions().getLen() <= 0 ) return;

	GLsizei size = factory->getPositions().getLen();

	float * hmem = new float[ size*3 ];
	float * dmem = (float*)factory->getPositions().map( MEM::MISC::BUF_CU );
	cudaMemcpy(hmem,dmem,sizeof(float)*3*size,cudaMemcpyDeviceToHost);
	factory->getPositions().unmap();

	cudaThreadSynchronize();

//        pr.use();

	glEnable(GL_LIGHTING);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);
	for( int i=0 ; i<size*3 ; i+=3 )
	{
		glPushMatrix();
		glColor3f(1,.5,0);
		glTranslatef( hmem[i] , hmem[i+1] , hmem[i+2] );
		glCallList( sphereListId );
		glPopMatrix();
	}
	glDisable(GL_COLOR_MATERIAL);
	glDisable(GL_LIGHTING);

//        Program::none();

	delete[]hmem;
}

