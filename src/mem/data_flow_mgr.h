#ifndef _DATA_FLOW_MGR_H_
#define _DATA_FLOW_MGR_H_

#include <GL/glew.h>

#include <string>

#include "ui/camera_manager.h"

namespace MEM
{
	namespace MISC
	{
		class GfxPlanetFactory;
		class PhxPlanetFactory;
	}

	/**
	 * @brief Klasa odpowiedzialna za wymianę danych między systemem plików,
	 * pamięcią programu oraz danymi przechowywanymi na karcie graficznej.
	 */
	class DataFlowMgr
	{
		public:
			DataFlowMgr();
			virtual ~DataFlowMgr();

			/**
			 * @brief Udostępnia dane planet istotne dla grafiki.
			 */
			MISC::GfxPlanetFactory *getGfxMem();
			/**
			 * @brief Udostępnia dane planet istotne dla fizyki.
			 */
			MISC::PhxPlanetFactory *getPhxMem();

			/**
			 * @brief Zapisuje aktualny stan symulacji.
			 *
			 * @param path Ścieżka do pliku, w którym ma zostać zapisana symulacja.
			 */
			void save( const std::string &path );
			/**
			 * @brief Wczytuje stan symulacji z pliku.
			 *
			 * @param path Ścieżka do pliku, z którego ma zostać wczytana symulacja.
			 */
			void load( const std::string &path );

			/**
			 * @brief Zapisuje stan symulacji w domyślnym pliku.
			 */
			void save();
			/**
			 * @brief Wczytuje stan symulacji z domyślnego pliku.
			 */
			void load();

			/**
			 * @brief Wczytuje materiały z pliku.
			 */
			GLuint loadMaterials();

			/**
			 * @brief Wczytuje tekstury z pliku.
			 */
			GLuint loadTextures();

			/**
			 * @brief Rejestruje obiekt kamery.
			 *
			 * @details Po rejestracji, kamera jest automatycznie zapisywana i
			 * wczytywana wraz z symulacją.
			 */
			void registerCam( UI::CameraMgr *cam );

		private:
			class Impl;
			Impl *impl;
	};
}

#endif // _DATA_FLOW_MGR_H_
