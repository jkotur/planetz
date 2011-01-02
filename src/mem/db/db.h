#ifndef _DB_H_
#define _DB_H_

#include <string>
#include <debug/routines.h>
#include "table_interface.h"

namespace MEM
{
	/**
	 * @brief Prosty interfejs do bazy danych.
	 *
	 * @details Ten interfejs pozwala na zapis całości danych do pliku lub odczyt całej
	 * tabeli z pliku. Do celów zapisu i odczytu symulacji jest to funkcjonalność wystarczająca.
	 */
	class Database
	{
		public:
			/**
			 * @brief Tworzy połączenie z bazą danych.
			 *
			 * @param cs Connection string - ciąg znaków zależny od użytej bazy.
			 */
			Database( const std::string &cs );

			/**
			 * @brief Zamyka połączenie z bazą danych.
			 */
			virtual ~Database();

			/**
			 * @brief Zapisuje tabelę w bazie.
			 *
			 * @details Tworzy nową tabelę, kasując starą, jeżeli istniała. Nowa tabela w
			 * bazie danych jest wypełniana danymi z tabeli przekazanej w argumencie.
			 *
			 * @param t Tabela, która ma zostać zapisana.
			 *
			 * @returns true, jeżeli operacja się powiodła, false w przeciwnym wypadku.
			 */
			bool save( const ITable& t );

			/**
			 * @brief Wczytuje tabelę z bazy.
			 *
			 * @param t Pusta tabela, do której mają zostać wczytane dane.
			 *
			 * @returns true, jeżeli operacja się powiodła, false w przeciwnym wypadku.
			 */
			bool load( ITable& t );

		protected:
			/**
			 * @brief Wysyła kod SQL odpowiedzialny za zapisanie tabeli.
			 *
			 * @param sql Kod SQL.
			 *
			 * @returns true, jeżeli operacja się powiodła, false w przeciwnym wypadku.
			 */
			virtual bool sendSaveString( const std::string& sql ) = 0;

			/**
			 * @brief Wysyła kod SQL odpowiedzialny za wyciągnięcie danych z bazy.
			 *
			 * @param sql Kod SQL.
			 *
			 * @param t Tabela, do której zostaną zapisane dane.
			 *
			 * @returns true, jeżeli operacja się powiodła, false w przeciwnym wypadku.
			 */
			virtual bool sendLoadString( const std::string& sql, ITable& t ) = 0;

			/**
			 * @brief Zależny od wybranego rodzaju bazy danych ciąg znaków,
			 * identyfikujący te bazę.
			 */
			const std::string connection_string;
	};
}

#endif // _DB_H 
