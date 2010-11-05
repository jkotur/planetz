#include "shader.h"

#include <sstream>
#include <fstream>

#include <cstdlib>

#include "debug/routines.h"

using namespace GFX;

Shader::Shader()
{
}

Shader::~Shader()
{
	glDeleteShader(_id);
}

ShaderManager::ShaderManager()
{
}

ShaderManager::~ShaderManager()
{
}

Shader* ShaderManager::loadShader( GLenum type , const std::string& path )
{
	ASSERT_MSG( type == GL_VERTEX_SHADER
	         || type == GL_FRAGMENT_SHADER 
		 || type == GL_GEOMETRY_SHADER
		  , "Unknown shader type specified");

	std::map<std::string,Shader*>::iterator i;
	if( (i=loaded_shaders.find(path)) != loaded_shaders.end() )
		return i->second;

	Shader* sh = new Shader();
	sh->_path = path;
	sh->_type = type;
	sh->_id = glCreateShader( type );
	std::string progs = readFile( path );
	const char*ps = progs.c_str();
	glShaderSource(sh->_id,1,&ps,NULL);
	glCompileShader(sh->_id);

	if( checkShaderLog( sh->_id , path ) )
		loaded_shaders[path] = sh;

	return sh;
}

bool ShaderManager::checkShaderLog( GLuint id , const std::string& path ) 
{                        
        int logLength = 0;
        int charsWritten  = 0;
        char *log;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH,&logLength);
        if( logLength > 1 ) {
                log = (char *)malloc(logLength);
                glGetShaderInfoLog(id, logLength, &charsWritten,log);
		log_printf(_ERROR,"Shader %s compile error:\n%s\n",path.c_str(),log);
                free(log);
		return false;
        } else	log_printf(INFO,"Shader %s loaded succesfully\n",path.c_str());
	return true;
}                        
                         


std::string ShaderManager::readFile( const std::string& path )
{
	std::ifstream file( path.c_str() );
	std::stringstream sstr;
	sstr << file.rdbuf();
	return sstr.str();
}

Program::Program( Shader*vs , Shader*fs , Shader*gs )
	: linked(false)  , _id(0) , vs(vs) , fs(fs) , gs(gs)
{
}

Program::~Program()
{
	glDeleteProgram(_id);
}

void Program::attach( const Shader* const sh )
{
	if(!_id ) _id = glCreateProgram();

	glAttachShader(_id,sh->id());

	linked = false;
	     if( sh->type() == GL_VERTEX_SHADER   )
		vs = sh;
	else if( sh->type() == GL_FRAGMENT_SHADER )
		fs = sh;
	else if( sh->type() == GL_GEOMETRY_SHADER )
		gs = sh;
}

void Program::link()
{
	if(!_id ) _id = glCreateProgram();

	glLinkProgram( _id );
	checkProgramLog( _id );
	linked = true;
}

void Program::use() const
{
	//if( !linked ) link();
	ASSERT_MSG( linked , "Program must be linked before use" )

	glUseProgram( _id );
}

bool Program::checkProgramLog( GLuint obj ) const
{       
        int infologLength = 0;
        int charsWritten  = 0;
        char *log;
        glGetProgramiv(obj, GL_INFO_LOG_LENGTH,&infologLength);
        if( infologLength > 1 ) {
                log = (char *)malloc(infologLength);
                glGetProgramInfoLog(obj, infologLength, &charsWritten, log);
		log_printf(_ERROR,"Program %d compiled with errors:\n%s\n",obj,log);
                free(log);          
		return false;
        } else	log_printf(INFO,"Program %d compiled succesfully\n");
	return true;
}       

void Program::none()
{
	glUseProgram(0);
}

