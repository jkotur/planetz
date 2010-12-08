#ifndef __MATERIALS_MANAGER_H__

#define __MATERIALS_MANAGER_H__

#include <GL/glew.h>

#include <vector>

namespace MEM
{
namespace MISC
{

struct Material 
{
	float r , g , b;
	float ke, ka, kd, ks;
	float alpha;
};

typedef std::vector<Material> Materials;

class MaterialsMgr {
public:
	MaterialsMgr( Materials*mat );
	virtual ~MaterialsMgr();

	unsigned int addMaterial();

	unsigned int addMaterial(
				float r , float g , float b ,
				float ke, float ka, float kd, float ks ,
				float alpha );

	void setColor3f( float , float , float );
	void setColor3i( int , int , int );

	void setKe( float );
	void setKa( float );
	void setKd( float );
	void setKs( float );
	void setAlpha( float );
private:
	unsigned int id;
	Materials*materials;
};

} // MISC
} // MEM

#endif /* __MATERIALS_MANAGER_H__ */

