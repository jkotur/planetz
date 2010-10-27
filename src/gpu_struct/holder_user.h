#ifndef HOLDER_USER_H
#define HOLDER_USER_H

#include "holder.h"

namespace CPU
{
	class MemMgr;
}

namespace GPU
{
	class HolderUser
	{
		public:
			HolderUser(uint32_t id);
			virtual ~HolderUser();

			friend class CPU::MemMgr;

		protected:
			static Holder* holder;
			uint32_t id;
	};
}

#endif // HOLDER_USER_H

