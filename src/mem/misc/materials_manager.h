#ifndef __MATERIALS_MANAGER_H__

#define __MATERIALS_MANAGER_H__

#include <GL/glew.h>

#include <vector>

namespace MEM
{
namespace MISC
{

/** 
 * @brief Struktura zawierająca wszystkie informacje o materiale planety
 */
struct Material 
{
	float r; /**< ilość czerwonego 0-1 */
	float g; /**< ilość zielonego  0-1 */
	float b; /**< ilość niebieskiego 0-1 */
	float ke; /**< ilość emitowangeo śiatła >0 */
	float ka; /**< reakcja na światło otoczenia  >0 */
	float kd; /**< reakcja na światło rozproszone >0 */
	float ks; /**< rekacja na światło odbite (nie używane) */
	float alpha; /**< przezroczystość */
	int   texture; /**< id tekstury */
	float ar; /**< czerowność atmosfery 0-1 */
	float ag; /**< zieloność atmosfery  0-1 */
	float ab; /**< niebieskość atmosfery 0-1 */
	float ad; /**< gęstość atmosfery (nie używane) */
	float al; /**< promień atmosfery, wyrażony wielokrotnością wzlgędem promienia planety */
};

typedef std::vector<Material> Materials;

/** 
 * @brief Klasa wspomagająca tworzenie zbioru materiałów.
 */
class MaterialsMgr {
public:
	/** 
	 * @brief Tworzy klasę na podstawie zbioru materiałów do którego
	 * mają być dodawane nowe materiały.
	 * 
	 * @param mat zbiór materiałów
	 */
	MaterialsMgr( Materials*mat );
	virtual ~MaterialsMgr();

	unsigned int addMaterial();

	unsigned int addMaterial(
				float r , float g , float b ,
				float ke, float ka, float kd, float ks ,
				float alpha ,
				int texture ,
				float ar, float ag, float ab, float ad, float al);

	void setColor3f( float , float , float );
	void setColor3i( int , int , int );

	void setKe( float );
	void setKa( float );
	void setKd( float );
	void setKs( float );
	void setAlpha( float );
	void setTexture( int );
	void setAtmosphere( float ar, float ag, float ab, float ad, float al );
private:
	unsigned int id;
	Materials*materials;
};

} // MISC
} // MEM

#endif /* __MATERIALS_MANAGER_H__ */

