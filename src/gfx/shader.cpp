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
	log_printf(DBG,"[DEL] Deleting shader  %s\n",_path.c_str());
	glDeleteShader(_id);
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

	Shader* sh = new Shader();
	sh->_path = path;
	sh->_type = type;
	sh->_id = glCreateShader( type );
	std::string progs = readFile( path );
	const char*ps = progs.c_str();
	glShaderSource(sh->_id,1,&ps,NULL);
	glCompileShader(sh->_id);

	// FIXME: or maybe add all shaders to one map? Fail is because
	//        warnings in shaders are treated same as errors :<
	if( checkShaderLog( sh->_id , path ) )
		loaded_shaders[path] = sh;
	else	loaded_with_errors.push_back( sh );

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
	: linked(false)  , _id(glCreateProgram()) , vs(vs) , fs(fs) , gs(gs)
{
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

void Program::link()
{
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

