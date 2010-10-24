#include <CEGUI.h>

#include "layout.h"

using namespace CEGUI;

Layout::Layout( const std::string& name )
{
	Window* myRoot = WindowManager::getSingleton().loadWindowLayout( name );
	System::getSingleton().setGUISheet( myRoot );
}

Layout::~Layout()
{
}

