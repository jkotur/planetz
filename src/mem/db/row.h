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
			virtual std::string getSaveString() const;

			/**
			 * @brief Zwraca kod SQL wczytujący wiersz z bazy.
			 *
			 * @returns Kod SQL.
			 */
			virtual std::string getLoadString() const;

			/**
			 * @brief Zwraca kod SQL tworzący tabelę - i usuwający poprzednią, jeżeli istniała.
			 */
			virtual std::string getCreationString() const;

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

		protected:
			/**
			 * @brief Zwraca alfanumeryczny ciąg znaków, będący nazwą tabeli.
			 *
			 * @returns Nazwa tabeli.
			 */
			virtual std::string getTableName() const = 0;

			/**
			 * @brief Zwraca nazwy kolumn w tabeli, oddzielone przecinkami.
			 *
			 * @returns Nazwy kolumn.
			 */
			virtual std::string getCellNames() const = 0;

			/**
			 * @brief Zwraca nazwy kolumn, wraz z ich typami w bazie.
			 *
			 * @returns Definicje kolumn.
			 */
			virtual std::string getCellDefs() const = 0;

			/**
			 * @brief Zwraca wartości wszystkich komórek wiersza.
			 *
			 * @returns Wartości oddzielone przecinkami.
			 */
			virtual std::string getCellValues() const = 0;
	};
}

#endif // _ROW_H_
