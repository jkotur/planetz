#ifndef _TABLE_H_
#define _TABLE_H_

#include <boost/foreach.hpp>
#include <string>
#include <sstream>
#include <list>
#include "row.h"

namespace MEM
{
	/**
	 * @brief Implementacja tabeli w bazie danych
	 */
	template<class RowType>
	class Table : public ITable
	{
		public:
			Table() {}
			virtual ~Table();

			virtual std::string getSaveString() const;
			virtual std::string getLoadString() const;
			virtual std::string getCreationString() const;

			/**
			 * @brief Dodaje wiersz do tabeli.
			 *
			 * @details Table zarządza czasem życia wszystkich wierszy - nie wolno
			 * ręcznie zwalniać pamięci wiersza, który został dodany do tabeli.
			 *
			 * @param row Wiersz, który ma być dodany.
			 */
			virtual void add( RowType *row );

			virtual Row* insert_new();

			/**
			 * @brief Kontener, używany do przechowywania wierszy tabeli.
			 */
			typedef std::list<RowType*> RowContainer;
			/**
			 * @brief Iterator po wierszach tabeli.
			 */
			typedef typename RowContainer::iterator iterator;
			/**
			 * @brief Stały iterator po wierszach tabeli.
			 *
			 * @details Stały iterator nie ma możliwości modyfikacji danych,
			 * po których iteruje
			 */
			typedef typename RowContainer::const_iterator const_iterator;
			/**
			 * @brief Liczba całkowita bez znaku.
			 */
			typedef typename RowContainer::size_type size_type;
			/**
			 * @brief Liczba całkowita ze znakiem.
			 */
			typedef typename RowContainer::difference_type difference_type;
			/**
			 * @brief Wskaźnik na element.
			 */
			typedef typename RowContainer::pointer pointer;
			/**
			 * @brief Referencja do elementu.
			 */
			typedef typename RowContainer::reference reference;

			/**
			 * @brief Zwraca iterator do pierwszego wiersza w tabeli.
			 */
			iterator begin();
			/**
			 * @brief Zwraca iterator do końca tabeli.
			 */
			iterator end();

			/**
			 * @brief Zwraca stały iterator do pierwszego wiersza w tabeli.
			 */
			const_iterator begin() const;
			/**
			 * @brief Zwraca stały iterator do końca tabeli.
			 */
			const_iterator end() const;

			/**
			 * @brief Czyści tabelę ze wszystkich wierszy.
			 */
			void clear();
			/**
			 * @brief Zwraca ilość wierszy w tabeli.
			 */
			size_type size() const;

		private:
			RowContainer rows;
			friend class Database;
			static const std::string transaction_begin;;
			static const std::string transaction_end;
	};

	template<class RowType>
	const std::string Table<RowType>::transaction_begin = "BEGIN TRANSACTION;\n";
	template<class RowType>
	const std::string Table<RowType>::transaction_end = "COMMIT;\n";

	template<class RowType>
	std::string Table<RowType>::getSaveString() const
	{
		log_printf(DBG, "gettin save string\n");
		std::stringstream ss;
		ss << transaction_begin;
		for( const_iterator i = begin(); i != end(); ++i )
		{
			ss << (*i)->getSaveString() << std::endl;
		}
		ss << transaction_end;
		log_printf(DBG, "gettin save string done: %s\n", ss.str().c_str());
		fflush(stdout);
		fflush(stderr);
		return ss.str();
	}

	template<class RowType>
	std::string Table<RowType>::getLoadString() const
	{
		// should be RowType::getLoadString, but C++ won't allow virtual static method :<
		return RowType().getLoadString();
	}

	template<class RowType>
	std::string Table<RowType>::getCreationString() const
	{
		// should be RowType::getCreationString, but C++ won't allow virtual static method :<
		return RowType().getCreationString();
	}

	template<class RowType>
	void Table<RowType>::add( RowType *row )
	{
		rows.push_back(row);
	}

	template<class RowType>
	Row* Table<RowType>::insert_new()
	{
		RowType *r = new RowType();
		add( r );
		return r;
	}

	template<class RowType>
	typename Table<RowType>::iterator Table<RowType>::begin()
	{
		return rows.begin();
	}

	template<class RowType>
	typename Table<RowType>::iterator Table<RowType>::end()
	{
		return rows.end();
	}

	template<class RowType>
	typename Table<RowType>::const_iterator Table<RowType>::begin() const
	{
		return rows.begin();
	}

	template<class RowType>
	typename Table<RowType>::const_iterator Table<RowType>::end() const
	{
		return rows.end();
	}

	template<class RowType>
	void Table<RowType>::clear()
	{
		for( const_iterator i = begin(); i != end(); ++i )
			delete *i;
		rows.clear();
	}

	template<class RowType>
	Table<RowType>::~Table<RowType>()
	{
		clear();
	}
	
	template<class RowType>
	typename Table<RowType>::size_type Table<RowType>::size() const
	{
		return rows.size();
	}
}

#endif // _TABLE_H_
