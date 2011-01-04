#ifndef _ROW_H_
#define _ROW_H_

#include <string>
#include "util/types.h"

namespace MEM
{
	/**
	 * @brief Klasa definiująca abstrakcyjny wiersz z bazy danych.
	 */
	class Row
	{
		public:
			Row();
			virtual ~Row();

			/**
			 * @brief Zwraca kod SQL zapisujący wiersz do bazy.
			 *
			 * @returns Kod SQL.
			 */
			virtual std::string getSaveString() const = 0;

			/**
			 * @brief Zwraca kod SQL wczytujący wiersz z bazy.
			 *
			 * @returns Kod SQL.
			 */
			virtual std::string getLoadString() const = 0;

			/**
			 * @brief Zwraca kod SQL tworzący tabelę - i usuwający poprzednią, jeżeli istniała.
			 */
			virtual std::string getCreationString() const = 0;

			/**
			 * @brief Rozmiar wiersza.
			 *
			 * @returns Liczba komórek w tym wierszu.
			 */
			virtual uint8_t size() const = 0;

			/**
			 * @brief Ustawia zawartość komórki na podstawie podanej wartości.
			 *
			 * @param idx Numer komórki, do której mają trafić dane.
			 *
			 * @param val Wartość do zapisu, w formie tekstowej.
			 */
			virtual void setCell( unsigned idx, const std::string& val ) = 0;
	};
}

#endif // _ROW_H_
