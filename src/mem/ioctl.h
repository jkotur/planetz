#ifndef _IOCTL_H_
#define _IOCTL_H_

namespace GPU
{
	class PhxPlanetFactory;
	class GfxPlanetFactory;
}

namespace MEM
{
	class IOCtl
	{
		public:
			IOCtl();
			virtual ~IOCtl();

			void save( PhxPlanetFactory*, GfxPlanetFactory* , const std::string &path );
			void load( PhxPlanetFactory*, GfxPlanetFactory* , const std::string &path );

			void save( PhxPlanetFactory*, GfxPlanetFactory* );
			void load( PhxPlanetFactory*, GfxPlanetFactory* );

		private:
			class Impl;
			Impl* impl;
	};
}

#endif // _IOCTL_H_
