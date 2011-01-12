#ifndef _MEM_SAVER_PARAMS_H_
#define _MEM_SAVER_PARAMS_H_

#include "holder.h"

#include "ui/camera_manager.h"

namespace MEM
{
namespace MISC
{
	/// @brief Klasa określająca zbiór danych potrzebnych do zapisu/odczytu symulacji.
	class SaverParams
	{
		public:
			/// @brief Inicjalizacja parametrów - ustawienie informacji o kamerze.
			SaverParams( UI::CameraMgr *cam);
			
			/// @brief Zwalnia pamięć po planet_info, jeżeli było ustawione.
			virtual ~SaverParams();

			/// @brief Informacje o planetach.
			CpuPlanetHolder *planet_info;

			/// @brief Informacja o kamerze.
			UI::CameraMgr *cam_info;
	};
}
}

#endif // _MEM_SAVER_PARAMS_H_

