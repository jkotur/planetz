#include "planetz_picker.h"

#include "constants.h"

using namespace GFX;

PlanetzPicker::PlanetzPicker( const MEM::MISC::GfxPlanetFactory * factory , int w , int h )
	: factory(factory) , w(w) , h(h)
	, vs(GL_VERTEX_SHADER  ,DATA("shaders/picker.vert"))
	, gs(GL_GEOMETRY_SHADER,DATA("shaders/picker.geom"))
	, fs(GL_FRAGMENT_SHADER,DATA("shaders/picker.frag"))
	, max(-1)
{
	vs.checkShaderLog();
	gs.checkShaderLog();
	fs.checkShaderLog();

	buffNames = new float[w*h];
	buffDepth = new float[w*h];

	pr.create(&vs,&fs,&gs,GL_POINTS,GL_QUAD_STRIP);

	radiusId = glGetAttribLocation( pr.id() , "radius" );
	namesId  = glGetAttribLocation( pr.id() , "name"   );

	glGenTextures( 1 , &colorTex );
	glBindTexture(GL_TEXTURE_2D, colorTex );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S    , GL_CLAMP  );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,BASE_W,BASE_H,0,GL_RGBA,GL_FLOAT,NULL);

	glGenTextures( 1, &depthTex );
	glBindTexture( GL_TEXTURE_2D, depthTex );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, BASE_W,BASE_H, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);

	glGenFramebuffers( 1, &fboId );
	glBindFramebuffer( GL_FRAMEBUFFER , fboId );

	glFramebufferTexture2D( GL_FRAMEBUFFER , GL_COLOR_ATTACHMENT0 ,
				GL_TEXTURE_2D , colorTex , 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER , GL_DEPTH_ATTACHMENT  ,
				GL_TEXTURE_2D , depthTex  , 0 );

	glBindFramebuffer( GL_FRAMEBUFFER , 0 );

}

PlanetzPicker::~PlanetzPicker()
{
	glDeleteTextures( 1 , &depthTex );
	glDeleteTextures( 1 , &colorTex );
	glDeleteFramebuffers( 1 , &fboId );
	delete[]buffNames;
	delete[]buffDepth;
}

void PlanetzPicker::render( int x , int y )
{
	resizeNames();

	max = -1;

//        glViewport( x-w/2 , y-h/2 , w , h );
//        glViewport( 0,0 , w , h );

	GLint viewport[4];

	glGetIntegerv (GL_VIEWPORT, viewport);

//        glShadeModel(GL_FLAT);

	glMatrixMode (GL_PROJECTION);
	glPushMatrix ();
	glLoadIdentity ();
	gluPickMatrix( x , viewport[3]-y , BASE_W , BASE_H , viewport);
	gluPerspective(75.0, (float)BASE_W/(float)BASE_H, 1, 10000);

	glMatrixMode(GL_MODELVIEW);
	
	glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);
	glClampColor(GL_CLAMP_READ_COLOR,GL_FALSE);
	glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);

	glEnableClientState( GL_VERTEX_ARRAY );
	glEnableVertexAttribArray( radiusId );
	glEnableVertexAttribArray( namesId  );

	factory->getPositions().bind();
	glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
	factory->getPositions().unbind();

	factory->getRadiuses().bind();
	glVertexAttribPointer( radiusId , 1, GL_FLOAT, GL_FALSE, 0, NULL );
	factory->getRadiuses().unbind();

	names.bind();
	glVertexAttribPointer( namesId , 1, GL_FLOAT  , GL_FALSE ,0, NULL );
	names.unbind();

	pr.use();
	glBindFramebuffer( GL_FRAMEBUFFER , fboId );

	glDrawBuffer( GL_COLOR_ATTACHMENT0 );
	glClearColor(.0,.0,.0,0);
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	glDrawArrays( GL_POINTS , 0 , factory->getPositions().getLen() );


//        glReadPixels(0,0,w,h,GL_RED            ,GL_FLOAT,buffNames);
//        glReadPixels(0,0,w,h,GL_DEPTH_COMPONENT,GL_FLOAT,buffDepth);
	glReadPixels(BASE_W/2-w/2,BASE_H/2-h/2,w,h,GL_RED            ,GL_FLOAT,buffNames);
	glReadPixels(BASE_W/2-w/2,BASE_H/2-h/2,w,h,GL_DEPTH_COMPONENT,GL_FLOAT,buffDepth);

	glBindFramebuffer( GL_FRAMEBUFFER , 0 );
	Program::none();

	glDisableVertexAttribArray( radiusId );
	glDisableVertexAttribArray( namesId  );
	glDisableClientState( GL_VERTEX_ARRAY );
	glMatrixMode (GL_PROJECTION);
	glPopMatrix();


//        for( int i=0 ; i<w*h ; i++ )
//                fprintf(stderr,"%.4f ",buffNames[i]);
//        fprintf(stderr,"\n");
//        for( int i=0 ; i<w*h ; i++ )
//                fprintf(stderr,"%.4f ",buffDepth[i]);
//        fprintf(stderr,"\n");
}

int PlanetzPicker::getId()
{
	if( max == -1 ) {
		max = 0;
		float d = 1.0f;
		for( int i=0 ; i<w*h ; i++ )
			if( buffDepth[i]<d )
				max = (int)(buffNames[i]+.5);
	}
	return max-1;
}

void PlanetzPicker::resizeNames()
{
	if( factory->getPositions().getLen() == names.getLen() ) return;

	names.resize( factory->getPositions().getLen() );

	float* pname = names.map( MEM::MISC::BUF_H );
	for( unsigned int i=0 ; i<names.getLen() ; i++ )
		pname[i] = i+1;
	names.unmap();
}

