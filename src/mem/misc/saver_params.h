#ifndef _MEM_SAVER_PARAMS_H_
#define _MEM_SAVER_PARAMS_H_

#include "holder.h"

class Camera;

namespace MEM
{
namespace MISC
{
	/// @brief Klasa określająca zbiór danych potrzebnych do zapisu/odczytu symulacji.
	class SaverParams
	{
		public:
			/// @brief Inicjalizacja parametrów - ustawienie informacji o kamerze.
			SaverParams(Camera *cam);
			
			/// @brief Zwalnia pamięć po planet_info, jeżeli było ustawione.
			virtual ~SaverParams();

			/// @brief Informacje o planetach.
			CpuPlanetHolder *planet_info;

			/// @brief Informacja o kamerze.
			Camera *cam_info;
	};
}
}

#endif // _MEM_SAVER_PARAMS_H_

