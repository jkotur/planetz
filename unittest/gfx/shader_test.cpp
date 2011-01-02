#include "shader_test.h"

#include <cppunit/extensions/HelperMacros.h>

#include <GL/glew.h>

#include "gfx/shader.h"
#include "constants.h"

#include "util/logger.h"

using namespace GFX;

void ShaderTest::setUp()
{
}

void ShaderTest::tearDown()
{
}

void ShaderTest::compileShadersTest()
{
	std::vector<Shader> shaders;

	shaders.push_back( Shader( GL_VERTEX_SHADER   , DATA("shaders/picker.vert") ) );
	shaders.push_back( Shader( GL_FRAGMENT_SHADER , DATA("shaders/picker.frag") ) );
	shaders.push_back( Shader( GL_GEOMETRY_SHADER , DATA("shaders/picker.geom") ) );
	shaders.push_back( Shader( GL_VERTEX_SHADER   , DATA("shaders/deffered_01.vert") ) );
	shaders.push_back( Shader( GL_FRAGMENT_SHADER , DATA("shaders/deffered_01.frag") ) );
	shaders.push_back( Shader( GL_GEOMETRY_SHADER , DATA("shaders/deffered_01.geom") ) );
	shaders.push_back( Shader( GL_VERTEX_SHADER   , DATA("shaders/deffered_02.vert") ) );
	shaders.push_back( Shader( GL_FRAGMENT_SHADER , DATA("shaders/deffered_02.frag") ) );
	shaders.push_back( Shader( GL_GEOMETRY_SHADER , DATA("shaders/deffered_02.geom") ) );
	shaders.push_back( Shader( GL_VERTEX_SHADER   , DATA("shaders/deffered_03.vert") ) );
	shaders.push_back( Shader( GL_FRAGMENT_SHADER , DATA("shaders/deffered_03.frag") ) );
	shaders.push_back( Shader( GL_GEOMETRY_SHADER , DATA("shaders/deffered_03.geom") ) );
	shaders.push_back( Shader( GL_VERTEX_SHADER   , DATA("shaders/deffered_04.vert") ) );
	shaders.push_back( Shader( GL_FRAGMENT_SHADER , DATA("shaders/deffered_04.frag") ) );
	shaders.push_back( Shader( GL_GEOMETRY_SHADER , DATA("shaders/deffered_04.geom") ) );
	shaders.push_back( Shader( GL_FRAGMENT_SHADER , DATA("shaders/deffered_05.frag") ) );

//        log_add(LOG_STREAM(stderr),LOG_PRINTER(vfprintf));

	for( std::vector<Shader>::iterator i = shaders.begin() ; i != shaders.end() ; ++i )
		CPPUNIT_ASSERT( i->checkShaderLog() );
	for( std::vector<Shader>::iterator i = shaders.begin() ; i != shaders.end() ; ++i )
		CPPUNIT_ASSERT( i->id() != 0 );
	for( std::vector<Shader>::iterator i = shaders.begin() ; i != shaders.end() ; ++i )
		log_printf(DBG,"%s: %d\n",i->path().c_str(),i->id());
}

void ShaderTest::linkProgramTest()
{
	Program pr(
		new Shader( GL_VERTEX_SHADER   , DATA("shaders/deffered_01.vert") ) ,
		new Shader( GL_FRAGMENT_SHADER , DATA("shaders/deffered_01.frag") ) ,
		new Shader( GL_GEOMETRY_SHADER , DATA("shaders/deffered_01.geom") ) );

	pr.geomParams(GL_POINTS,GL_QUADS);

	CPPUNIT_ASSERT( pr.link() );
}

