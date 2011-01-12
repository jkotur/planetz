#include "camera_manager.h"

#include <GL/glew.h>

#include "util/logger.h"

#include "debug/routines.h"

using namespace UI;

CameraMgr::CameraMgr(
		Vector3 p ,
		Vector3 l ,
		Vector3 u )
{
	log_printf(DBG,"CameraMgr is borning\n");
	cams[FREELOOK] = new CamFreeLook();
	cams[LOCKED  ] = new CamLocked();
	cams[ZOOMIN  ] = new CamZoomIn();

	currmat = new float[16];

	set_perspective( p , l , u );

	clear();
}

void CameraMgr::init()
{
	emit_sig();
}

void CameraMgr::clear()
{
	next.clear();

	currcam = cams[0];
	float defaultspd = .2f;
	currcam->born( currmat , (void*)&defaultspd );
}

CameraMgr::~CameraMgr()
{
	for( unsigned i=0 ; i<CAMERA_NUM ; ++i )
		delete cams[i];
	delete currmat;
}

void CameraMgr::request( enum CAMERA_TYPES type , void*data )
{
	next.push_back( std::make_pair( cams[(unsigned)type] , data ) );
}

void CameraMgr::update( enum CAMERA_TYPES type , void*data )
{
	cams[(unsigned)type]->learn(data);
}

void CameraMgr::set_perspective(
	 const Vector3& pos 
	,const Vector3& lookat 
	,const Vector3& up )
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	gluLookAt(
		pos.x , pos.y , pos.z ,
		lookat.x , lookat.y , lookat.z ,
		up.x , up.y , up.z
	);
	glGetFloatv( GL_MODELVIEW_MATRIX , currmat );
	glPopMatrix();
}

void CameraMgr::draw() const
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glLoadMatrixf( currmat );
	glGetFloatv( GL_MODELVIEW_MATRIX , currmat );
	glPopMatrix();
}

void CameraMgr::signal()
{
	currmat = currcam->work();

	swap_camera();

	if( !currmat ) {
		request(FREELOOK);
		swap_camera();
		ASSERT_MSG( currmat , "currmat should never be NULL");
	}

	emit_sig();
}

void CameraMgr::swap_camera()
{
	if( !next.size() ) return;

	if( next.front().first == currcam ) {
		CameraQueuePair cp = next.front();
		next.pop_front();
		currcam->learn(cp.second);
		return;
	}

	float* m = currcam->die();

	if( !m ) return;

	CameraQueuePair cp = next.front();
	next.pop_front();

	currmat = m;

	currcam = cp.first;
	currcam->born( currmat , cp.second );
}

void CameraMgr::emit_sig()
{
	float m[16];
	memcpy(m,currmat,12*sizeof(float));
	m[12] = m[13] = m[14] = 0.0f;
	m[15] = 1.0f;

	sigCamChanged( m );
}

void CameraMgr::on_key_down( SDLKey a , Uint16 b , Uint8 c )
{	currcam->on_key_down(a,b,c); }

void CameraMgr::on_mouse_motion( int x , int y )
{	currcam->on_mouse_motion(x,y); }

void CameraMgr::on_button_up( int a , int b , int c )
{	currcam->on_button_up( a , b , c ); }

bool CameraMgr::on_button_down( int a , int b , int c )
{	return currcam->on_button_down( a , b, c ); }

