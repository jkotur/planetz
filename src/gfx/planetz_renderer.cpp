#include "planetz_renderer.h"

#include <GL/glew.h>

#include "sphere/sphere.h"
#include "util/vector.h"

#include "constants.h"

using namespace GFX;

PlanetzRenderer::PlanetzRenderer( const GPU::GfxPlanetFactory * factory )
	: texModelId(0) , factory( factory ) 
{
}

PlanetzRenderer::~PlanetzRenderer()
{
	log_printf(DBG,"[DEL] Deleting PlanetzRenderer\n");
}

void PlanetzRenderer::setModels( GPU::PlanetzModel mod )
{
	modPlanet = mod;
}

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

void PlanetzRenderer::prepare()
{
	ShaderManager shm;

	Shader*vs = shm.loadShader(GL_VERTEX_SHADER,DATA("shaders/shader.vert"));
	Shader*fs = shm.loadShader(GL_FRAGMENT_SHADER,DATA("shaders/shader.frag"));
	Shader*gs = shm.loadShader(GL_GEOMETRY_SHADER,DATA("shaders/shader.geom"));

	pr.attach( vs );
	pr.attach( fs );
	pr.attach( gs );
	pr.geomParams( GL_POINTS , GL_TRIANGLES );

	pr.link();

	texModelId = glGetUniformLocation( pr.id() , "models" );

	sphereListId = glGenLists(1);
	glNewList(sphereListId,GL_COMPILE);
	draw_sphere_2(Sphere::get_obj(5));
	glEndList();

}

void PlanetzRenderer::draw_calllist() const
{
	if( factory->getPositions().getLen() <= 0 ) return;

	GLsizei size = factory->getPositions().getLen();

	float * hmem = new float[ size*3 ];
	float * dmem = (float*)factory->getPositions().map( GPU::BUF_CU );
	cudaMemcpy(hmem,dmem,sizeof(float)*3*size,cudaMemcpyDeviceToHost);
	factory->getPositions().unmap();

	cudaThreadSynchronize();

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

	delete[]hmem;
}

void PlanetzRenderer::draw_geomshader() const
{
	ASSERT_MSG( modPlanet.vertices , "Before drawing, models texture id must be specified by calling setModels" );

	glPointSize( 3 );
	glEnableClientState( GL_VERTEX_ARRAY );

	glActiveTexture(GL_TEXTURE0);

	pr.use();
	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );

	for( int i=0 ; i<modPlanet.parts ; i++ )
	{
		glBindTexture(GL_TEXTURE_1D,modPlanet.vertices[i]);
		glUniform1i(texModelId,0);

		glDrawArrays( GL_POINTS , 0 , factory->getPositions().getLen() );
	}

	factory->getPositions().unbind();
	Program::none();

	glBindTexture(GL_TEXTURE_1D,0);

	glDisableClientState( GL_VERTEX_ARRAY );
}

void PlanetzRenderer::draw() const
{
	draw_calllist();
}

