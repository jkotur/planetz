#ifndef __SHADER_H__

#define __SHADER_H__

#include <GL/glew.h>

#include <string>
#include <map>
#include <list>


namespace GFX
{

/** 
 * @brief Klasa reprezentująca shadera. Czyta go z pliku i kompiluje.
 */
class Shader {
public:
	/** 
	 * @brief Kompiluje shader na podstawie pliku.
	 * 
	 * @param type typ shadera. Dostępne w zależnosci od wersji opengla
	 * @param path ścieżka do pliku z shaderem
	 */
	Shader( GLenum type , const std::string& path );
	/** 
	 * @brief Kasuje shader z pamięci opengla
	 */
	virtual ~Shader();

	     GLuint  id  () const { return _id;   } /**< Zwraca id shadera */
	     GLenum  type() const { return _type; } /**< Zwraca typ shadera */
	std::string  path() const { return _path; } /**< Zwraca ścieżkę do shadera */
	
	/** 
	 * @brief Sprawdza czy przy kompilacji shadera nie wystąpiły błędy
	 * 
	 * @return true jeśli nie, false wpp
	 */
	bool checkShaderLog();
private:
	std::string readFile( const std::string& path );

	     GLuint _id  ;
	     GLenum _type;
	std::string _path;
};

/** 
 * @brief Klasa odpowiedzialna za ładownie shaderów do pamięci.
 * Dodatkowo pamięta shadery, tak że może zwrócić kopię już załadowanego,
 * zamiast ładować dwa razy.
 */
class ShaderManager {
public:
	/** 
	 * @brief Konstruuje pustego menadżera
	 */
	ShaderManager ();
	/** 
	 * @brief Kasuje wszystkie shadery załadowane przez tego menadżera
	 */
	virtual ~ShaderManager();

	/** 
	 * @brief Ładuje nowy shader z pamięci, bądź zwraca kopię już załadowanego
	 * 
	 * @param type typ shadera do załadowania
	 * @param path ścieżka do pliku z kodem shadera
	 * 
	 * @return zwraca obiekt reprezentujący shader
	 */
	Shader*loadShader( GLenum type , const std::string& path );
private:

	std::map<std::string,Shader*> loaded_shaders;
	std::list<Shader*> loaded_with_errors;
};

/** 
 * @brief Klasa reprezentująca program opengla na GPU. 
 * Zgodna z OpenGL 3.2. Wspiera tylko shadery wierzchołków,
 * fragmentów i geometrii.
 */
class Program {
public:
	/** 
	 * @brief Tworzy program na podstawie podanych shaderów.
	 * 
	 * @param vs shader wierzchołków
	 * @param fs shader fragmentów
	 * @param gs shader geometrii
	 */
	Program( Shader*vs = NULL , Shader*fs = NULL , Shader*gs = NULL );
	/** 
	 * @brief Kasuje program z pamięci opengla
	 */
	virtual ~Program();

	/** 
	 * @brief Tworzy program na podstawie shadera wierzchołków
	 * i ewentualnie na podstawie shadera fragmentów.
	 * 
	 * @param vs shader wierzchołków
	 * @param fs shader fragmentów
	 */
	void create( Shader*vs , Shader*fs = NULL );
	/** 
	 * @brief Tworzy największy program, który zawiera wszystkie shadery.
	 * 
	 * @param vs shader wierzchołków
	 * @param fs shader fragmentów
	 * @param gs shader geometrii
	 * @param in typ geometrii shadera geometrii na wejściu.
	 * Zgodny z api opengla.
	 * @param out typ geometrii shadera geometrii na wyjściu.
	 * Zgodny z api opengla.
	 */
	void create( Shader*vs , Shader*fs , Shader*gs , const GLenum in , const GLenum out );

	GLuint id() const { return _id; } /**< Zwraca id programu */

	/** 
	 * @brief Dodaje lub podmienia shader w programie.
	 * Po wywołaniu tej funkcji shader wymaga przelinkowania.
	 * 
	 * @param sh shader do dołączenia.
	 */
	void attach( const Shader* const sh );
	/** 
	 * @brief Linkuje program, tak aby był gotowy do użycia w openglu.
	 *
	 * @return true jeśli linkowanie wypadnie poprawnie, false wpp
	 */
	bool link();
	/** 
	 * @brief Nakazuje openglowi używania danego programu od tej chwili
	 */
	void use() const;

	/** 
	 * @brief Ustawia geometrję wejściową i wyjściową shadera geometrii.
	 * Bez wywołania tej funkcji, program z shaderem geometrii nie zostanie
	 * poprawnie zlinkowany.
	 * 
	 * @param in typ geometrii shadera geometrii na wejściu.
	 * Zgodny z api opengla.
	 * @param out typ geometrii shadera geometrii na wyjściu.
	 * Zgodny z api opengla.
	 */
	void geomParams( GLenum in , GLenum out );

	/** 
	 * @brief Ustawia pusty program jako aktualny (domyślny pipeline)
	 */
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

