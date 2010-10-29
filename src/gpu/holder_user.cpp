#include "holder_user.h"

using namespace GPU;

HolderUser::holder = NULL;

HolderUser::HolderUser(uint32_t _id)
	: id( _id )
{
	assert( holder );
}

HolderUser::~HolderUser()
{
}
