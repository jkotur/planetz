#include "texture.h"

#include <SDL/SDL_image.h>

#include "../util/logger.h"

using namespace GFX;

Texture::~Texture()
{
	log_printf(DBG,"[DEL] Deleting texture %s\n",path.c_str());
	glDeleteTextures(1,&tex);
}

void Texture::bind() const
{
	glBindTexture(GL_TEXTURE_2D,tex);
}

void Texture::unbind()
{
	glBindTexture(GL_TEXTURE_2D,0);
}

TextureManager::TextureManager()
{
}

TextureManager::~TextureManager()
{
	for( std::map<string,Texture*>::iterator i=loaded_textures.begin() ; i!=loaded_textures.end() ; ++i ) 
		delete i->second;
}

Texture* TextureManager::loadTexture( const string& file )
{
	return loadTexture(file.c_str());
}

Texture* TextureManager::loadTexture( const char* file )
{
	std::map<std::string,Texture*>::iterator i;
	if( (i=loaded_textures.find(file)) != loaded_textures.end() )
		return i->second;

	SDL_Surface* surface = IMG_Load(file);

	if( !surface ) {
		log_printf(DBG,"SDL could not load image: %s\n", SDL_GetError());
		SDL_Quit();
		return NULL;
	}
	GLuint texture;
	glPixelStorei(GL_UNPACK_ALIGNMENT,4);
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	SDL_PixelFormat *format = surface->format;
	if (format->Amask) 
		gluBuild2DMipmaps(GL_TEXTURE_2D, 4,
				surface->w, surface->h,
				GL_RGBA,GL_UNSIGNED_BYTE, surface->pixels);
	else	gluBuild2DMipmaps(GL_TEXTURE_2D, 3,
				surface->w, surface->h,
				GL_RGB, GL_UNSIGNED_BYTE, surface->pixels);

	Texture*t = new Texture(file,texture);
	loaded_textures[file] = t;

	log_printf(INFO,"Texture %s loaded succesfully\n",file);

	SDL_FreeSurface(surface);

	return t;
}

