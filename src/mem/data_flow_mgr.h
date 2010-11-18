#ifndef _DATA_FLOW_MGR_H_
#define _DATA_FLOW_MGR_H_

#include <string>

namespace MEM
{
	namespace MISC
	{
		class GfxPlanetFactory;
		class PhxPlanetFactory;
	}

	class DataFlowMgr
	{
		public:
			DataFlowMgr();
			virtual ~DataFlowMgr();

			MISC::GfxPlanetFactory *getGfxMem();
			MISC::PhxPlanetFactory *getPhxMem();

			void save( const std::string &path );
			void load( const std::string &path );

			void save();
			void load();

		private:
			class Impl;
			Impl *impl;
	};
}

#endif // _DATA_FLOW_MGR_H_
