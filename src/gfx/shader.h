#ifndef __SHADER_H__

#define __SHADER_H__

#include <GL/glew.h>

#include <string>
#include <map>


namespace GFX
{

class Shader {
	friend class ShaderManager;

	Shader();
	virtual ~Shader();
public:
	     GLuint  id  () const { return _id;   }
	     GLenum  type() const { return _type; }
	std::string  path() const { return _path; }
	
private:
	     GLuint _id  ;
	     GLenum _type;
	std::string _path;
};

class ShaderManager {
public:
	ShaderManager ();
	virtual ~ShaderManager();

	Shader*loadShader( GLenum type , const std::string& path );
private:

	std::string readFile( const std::string& path );
	bool checkShaderLog( GLuint id , const std::string& path );

	std::map<std::string,Shader*> loaded_shaders;
};

class Program {
public:
	Program( Shader*vs = NULL , Shader*fs = NULL , Shader*gs = NULL );
	virtual ~Program();

	void create( Shader*vs , Shader*fs = NULL );
	void create( Shader*vs , Shader*fs , Shader*gs , const GLenum in , const GLenum out );

	GLuint id() const { return _id; }

	void attach( const Shader* const sh );
	void link();
	void use() const;

	void geomParams( GLenum in , GLenum out );

	static void none();
private:
	bool checkProgramLog( GLuint obj ) const;

	bool linked;

	GLuint _id;

	const Shader* vs;
	const Shader* fs;
	const Shader* gs;
};

} // GFX


#endif /* __SHADER_H__ */

