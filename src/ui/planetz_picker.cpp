#include "planetz_picker.h"

#include "constants.h"

using namespace UI;
using namespace GFX;

PlanetzPicker::PlanetzPicker( const MEM::MISC::GfxPlanetFactory * factory , int w , int h , int winw , int winh )
	: factory(factory) , w(w) , h(h) , winw(winw) , winh(winh)
	, vs(GL_VERTEX_SHADER  ,DATA("shaders/picker.vert"))
	, gs(GL_GEOMETRY_SHADER,DATA("shaders/picker.geom"))
	, fs(GL_FRAGMENT_SHADER,DATA("shaders/picker.frag"))
	, max(0u)
{
	vs.checkShaderLog();
	gs.checkShaderLog();
	fs.checkShaderLog();

	buffNames = new uint32_t[w*h];
	buffDepth = new float[w*h];

	pr.create(&vs,&fs,&gs,GL_POINTS,GL_QUAD_STRIP);

	radiusId = glGetAttribLocation( pr.id() , "radius" );
	namesId  = glGetAttribLocation( pr.id() , "name"   );

	// is this or bind needed? 
//        outloc   = glGetFragDataLocation( pr.id() , "ints" );

	sphereTex = generate_sphere_texture( 128 , 128 );
	sphereTexId = glGetUniformLocation( pr.id() , "sphere" );

	glUniform1i( sphereTexId , 0 );

	glGenTextures( 1 , &colorTex );
	glGenTextures( 1 , &depthTex );

	generate_fb_textures();

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
	glDeleteTextures( 1 , &sphereTex );
	glDeleteTextures( 1 , &depthTex );
	glDeleteTextures( 1 , &colorTex );
	glDeleteFramebuffers( 1 , &fboId );
	delete[]buffNames;
	delete[]buffDepth;
}

void PlanetzPicker::resize( int w , int h )
{
	winw = w ;
	winh = h ;
	generate_fb_textures();
}

bool PlanetzPicker::on_button_down( int b , int x , int y )
{
	if( b != PICK_BUTTON ) return false;

	render( x , y );

	sigPlanetPicked( getId() );

	return false;
}

void PlanetzPicker::generate_fb_textures()
{
	glBindTexture(GL_TEXTURE_2D, colorTex );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S    , GL_CLAMP  );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32UI,winw,winh,0,GL_RGBA_INTEGER,GL_UNSIGNED_INT,NULL);

	glBindTexture( GL_TEXTURE_2D, depthTex );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, winw,winh, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
}

GLuint PlanetzPicker::generate_sphere_texture( int w , int h )
{
	float*sphere = new float[ w * h ];

	float w2 = (float)w/2.0f;
	float h2 = (float)h/2.0f;

	for( int wi=0 ; wi<w ; wi++ )
		for( int hi=0 ; hi<h ; hi++ )
		{
			float x = ((float)wi - w2)/(float)w2;
			float y = ((float)hi - h2)/(float)h2;
			float a = 1 <= x*x + y*y ? 0.0 : 1.0 ;

			sphere[ wi + hi*w ] = a;
		}

	GLuint texId;
	glGenTextures( 1 , &texId );
	glBindTexture(GL_TEXTURE_2D, texId );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S    , GL_CLAMP  );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB16F,w,h,0,GL_RED,GL_FLOAT,sphere);
	delete[]sphere;
	return texId;
}

void PlanetzPicker::render( int x , int y )
{
	resizeNames();

	max = 0u;

//        glViewport( x-w/2 , y-h/2 , w , h );
//        glViewport( 0,0 , w , h );

	GLint viewport[4];

	glGetIntegerv (GL_VIEWPORT, viewport);

//        glShadeModel(GL_FLAT);
	glEnable( GL_DEPTH_TEST );

	glMatrixMode (GL_PROJECTION);
	glPushMatrix ();
	glLoadIdentity ();
	gluPickMatrix( x , viewport[3]-y , winw , winh , viewport);
	gluPerspective(75.0, (float)winw/(float)winh, 1, VIEW_LENGHT );

	glMatrixMode(GL_MODELVIEW);
	
//        glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);
//        glClampColor(GL_CLAMP_READ_COLOR,GL_FALSE);
//        glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);

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
	glVertexAttribIPointer( namesId , 1, GL_UNSIGNED_INT ,0, NULL );
	names.unbind();

	pr.use();
	glBindFramebuffer( GL_FRAMEBUFFER , fboId );

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D , sphereTex );

	glDrawBuffer( GL_COLOR_ATTACHMENT0 );
	glReadBuffer( GL_COLOR_ATTACHMENT0 );

	glClearColor(0,0,0,0);
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	glDrawArrays( GL_POINTS , 0 , factory->size() );

	memset(buffNames,1,sizeof(uint32_t)*w*h);
	glReadPixels(winw/2-w/2,winh/2-h/2,w,h,GL_RED_INTEGER    ,GL_UNSIGNED_INT,buffNames);
	glReadPixels(winw/2-w/2,winh/2-h/2,w,h,GL_DEPTH_COMPONENT,GL_FLOAT,buffDepth);

//        fprintf(stderr,"%d:%d  %d:%d  ",winw/2-w/2,winh/2-h/2,w,h);
//        for( int i=0 ; i<w*h ; i++ )
//                fprintf(stderr,"%d ",buffNames[i]);
//        fprintf(stderr,"\n");
	glBindTexture( GL_TEXTURE_2D , 0 );

	glBindFramebuffer( GL_FRAMEBUFFER , 0 );
	Program::none();

	glDisableVertexAttribArray( radiusId );
	glDisableVertexAttribArray( namesId  );
	glDisableClientState( GL_VERTEX_ARRAY );
	glMatrixMode (GL_PROJECTION);
	glPopMatrix();

	glDisable( GL_ALPHA_TEST );
	glDisable( GL_DEPTH_TEST );
}

int PlanetzPicker::getId()
{
	if( max == 0u ) {
		max = 0;
		float d = 1.0f;
		for( int i=0 ; i<w*h ; i++ )
			if( buffDepth[i]<d )
				max = buffNames[i];
	}
	ASSERT( max <= factory->size() );
	return max <= factory->size() ? max-1 : -1;
}

void PlanetzPicker::resizeNames()
{
	if( factory->size() == names.getLen() ) return;

	names.resize( factory->size() );

	uint32_t* pname = names.map( MEM::MISC::BUF_H );
	for( unsigned int i=0 ; i<names.getLen() ; i++ )
		pname[i] = i+1;
	names.unmap();
}

