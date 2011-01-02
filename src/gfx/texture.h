/**
 * @file texture.h
 * @author Jakub Kotur 
 * @version 0.1
 *
 * @section LICENSE
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of
 * the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details at
 * http://www.gnu.org/copyleft/gpl.html
 *
 * @section DESCRIPTION
 *
 * Ladowanie oraz bindowanie tekstur z formatow obslugiwanych przez
 * SDL_image
 */

#ifndef __TEXTURE_H__

#define __TEXTURE_H__

#include <GL/glew.h>

#include <string>
#include <map>

using std::string;

namespace GFX {
	
/** 
 * @brief Klasa reprezentująca teksturę. Może być zbindowana do api opengla.
 * Za konstrukcję i dekstrukcję jej instancji odpowiada TextureManager.
 */
class Texture {
	friend class TextureManager;

	Texture( string _path , GLuint _tex ) : path(_path) , tex(_tex) {}
	virtual ~Texture();
public:
	/** 
	 * @brief binduje teksturę jako GL_TEXTURE_2D do api opengla,
	 * w aktulanie aktywnej teksturze
	 */
	void bind() const;
	/** 
	 * @brief binduję pustą teksturę do GL_TEXTURE_2D
	 */
	static void unbind();
private:
	string path;
	GLuint tex;
};

/** 
 * Klasa odpowiedzialan za ładownie tekstur 2D z plików. Wszystkie tekstury
 * są mipmapowane i mają bilinearne mapowanie. 
 * Zawiera optymalizację polegającą na agregowaniu już załadownych
 * tekstur, i zwracaniu ich gdy były już ładowane.
 */
class TextureManager {
public:
	/** 
	 * @brief Konstruuje pustego menadżera tekstur.
	 */
	TextureManager();
	/** 
	 * @brief Kasuje wszystkie załadowne tekstury z pamięci karty graficznej.
	 */
	virtual ~TextureManager();

	/** 
	 * @brief Ładuje z pliku nową teksturę
	 * 
	 * @param str ścieżka względem binarki lub bezwzgędne do pliku z obrazem
	 * 
	 * @return wkaśnik na teksturę
	 */
	Texture* loadTexture( const string& str );
	/** 
	 * @brief Analogincznie tylko przyjmue c-stinga
	 * 
	 * @param str ścieżka względem binarki lub bezwzgędne do pliku z obrazem
	 * 
	 * @return wkaśnik na teksturę
	 */
	Texture* loadTexture( const char* str );
	// TODO: unloading texutres
private:
	std::map<string,Texture*> loaded_textures;
};


}

#endif /* __TEXTURE_H__ */

