#ifndef _TABLE_INTERFACE_H_
#define _TABLE_INTERFACE_H_

#include <string>

namespace MEM
{
	class Row;

	/**
	 * @brief Interfejs tabeli w bazie danych.
	 */
	class ITable
	{
		public:
			/**
			 * @brief Zwraca kod SQL zapisujący tabelę w bazie danych.
			 *
			 * @returns Kod SQL.
			 */
			virtual std::string getSaveString() const = 0;
			/**
			 * @brief Zwraca kod SQL pobierający tabelę z bazy danych.
			 *
			 * @returns Kod SQL.
			 */
			virtual std::string getLoadString() const = 0;
			/**
			 * @brief Zwraca kod SQL tworzący tabelę w bazie danych.
			 *
			 * @returns Kod SQL.
			 */
			virtual std::string getCreationString() const = 0;

			/**
			 * @brief Wstawia nowy wiersz do tabeli
			 *
			 * @returns Wskaźnik na nowy wiersz.
			 */
			virtual Row* insert_new() = 0;
	};
}

#endif // _TABLE_INTERFACE_H_

