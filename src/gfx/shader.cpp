#include "shader.h"

#include <sstream>
#include <fstream>

#include <cstdlib>

#include "debug/routines.h"

using namespace GFX;

Shader::Shader( GLenum type , const std::string& path )
{
	_path = path;
	_type = type;
	_id = glCreateShader( type );
	std::string progs = readFile( path );
	const char*ps = progs.c_str();
	glShaderSource(_id,1,&ps,NULL);
	glCompileShader(_id);
}

Shader::~Shader()
{
	log_printf(DBG,"[DEL] Deleting shader  %s\n",_path.c_str());
	glDeleteShader(_id);
}

std::string Shader::readFile( const std::string& path )
{
	std::ifstream file( path.c_str() );
	std::stringstream sstr;
	sstr << file.rdbuf();
	return sstr.str();
}

bool Shader::checkShaderLog()
{
        int logLength = 0;
        int charsWritten  = 0;
        char *log;
        glGetShaderiv(_id, GL_INFO_LOG_LENGTH,&logLength);
	GLint stat;
	glGetShaderiv(_id, GL_COMPILE_STATUS,&stat);
//        log_printf(INFO,"Shader status: %s\n",stat?"OK":"FAIL");
	if( logLength > 1 ) { // verbose mode
//        if(!stat ) { // quiet mode 
                log = (char *)malloc(logLength);
                glGetShaderInfoLog(_id, logLength, &charsWritten,log);
		log_printf(stat?_WARNING:_ERROR,"Shader %s compile %s:\n%s\n",_path.c_str(),stat?"warnings":"errors",log);
                free(log);
		return stat;
        } else	log_printf(INFO,"Shader %s compiled succesfully\n",_path.c_str());
	return stat;
}                        
                         
ShaderManager::ShaderManager()
{
}

ShaderManager::~ShaderManager()
{
	for( std::map<std::string,Shader*>::iterator i = loaded_shaders.begin() ; i!=loaded_shaders.end() ; ++i )
		delete i->second;
	for( std::list<Shader*>::iterator i = loaded_with_errors.begin() ; i!= loaded_with_errors.end() ; ++i )
		delete *i;
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

	Shader* sh = new Shader( type , path );

	// FIXME: or maybe add all shaders to one map? Fail is because
	//        warnings in shaders are treated same as errors :<
	if( sh->checkShaderLog() )
		loaded_shaders[path] = sh;
	else	loaded_with_errors.push_back( sh );

	return sh;
}

Program::Program( Shader*vs , Shader*fs , Shader*gs )
	: linked(false)  , _id(glCreateProgram())
{
	attach( vs );
	attach( fs );
	attach( gs );
}

Program::~Program()
{
	glDeleteProgram(_id);
}

void Program::create( Shader*vs , Shader*fs )
{
	attach( vs );
	attach( fs );
	
	link();
}

void Program::create( Shader*vs , Shader*fs , Shader*gs , const GLenum in , const GLenum out )
{
	attach( vs );
	attach( fs );
	attach( gs );

	geomParams( in , out );

	link();
}

void Program::attach( const Shader* const sh )
{
	glAttachShader(_id,sh->id());

	linked = false;
	     if( sh->type() == GL_VERTEX_SHADER   )
		vs = sh;
	else if( sh->type() == GL_FRAGMENT_SHADER )
		fs = sh;
	else if( sh->type() == GL_GEOMETRY_SHADER ) {
		gs = sh;
		int temp;
		glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT,&temp);
		glProgramParameteriEXT(_id,GL_GEOMETRY_VERTICES_OUT_EXT,temp);
	}

}

void Program::geomParams( GLenum in , GLenum out )
{
	glProgramParameteriEXT(_id,GL_GEOMETRY_INPUT_TYPE_EXT ,in );
	glProgramParameteriEXT(_id,GL_GEOMETRY_OUTPUT_TYPE_EXT,out);
}

bool Program::link()
{
	glLinkProgram( _id );
	linked = checkProgramLog( _id );
	return linked;
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
	GLint stat;
	glGetProgramiv(obj, GL_LINK_STATUS , &stat );
	if(!stat ) {
//        if( infologLength > 1 ) {
                log = (char *)malloc(infologLength);
                glGetProgramInfoLog(obj, infologLength, &charsWritten, log);
		log_printf(_ERROR,"Program %d linked with errors:\n%s\n",obj,log);
                free(log);          
		return false;
        } else	log_printf(INFO,"Program %d linked succesfully\n");
	return stat;
}       

void Program::none()
{
	glUseProgram(0);
}

