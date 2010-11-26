#ifndef __MATERIALS_MANAGER_H__

#define __MATERIALS_MANAGER_H__

#include <GL/glew.h>

#include <list>

namespace MEM
{
class MaterialsMgr {
	struct Material 
	{
		float r , g , b;
		float ke, ka, kd, ks;
		float alpha;
	};
public:
	MaterialsMgr ();
	virtual ~MaterialsMgr();

	unsigned int addMaterial();

	unsigned int addMaterial( float r , float g , float b ,
	                       float ke, float ka, float kd, float ks ,
			       float alpha );

	void setColor3f( float , float , float );
	void setColor3i( int , int , int );

	void setKe( float );
	void setKa( float );
	void setKd( float );
	void setKs( float );
	void setAlpha( float );

	GLuint compile();

	GLuint getTex()
	{	return texId; }
	
private:
	unsigned int id;
	GLuint texId;
	std::list<Material> materials;
};

} // MEM

#endif /* __MATERIALS_MANAGER_H__ */

