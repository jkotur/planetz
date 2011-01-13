#ifndef __PLANETZ_PICKER_H__

#define __PLANETZ_PICKER_H__

#include <boost/signal.hpp>

#include "mem/misc/gfx_planet_factory.h"
#include "mem/misc/buffer.h"
#include "gfx/shader.h"
#include "ui/input_listener.h"

namespace UI
{
/** 
 * @brief Klasa odpowiedzialna za sprawdzanie która planeta jest w okolicy myszki.
 *
 * Działa ona poprzez renderowanie planet w dodatkowym buforze, i sprawdznie która
 * planeta, w okolicy myszki, znajduje się najbliżej kamery.
 */
class PlanetzPicker : public InputListener {
public:
	/** 
	 * @brief Tworzy klasę i przygotowuje ją do działania.
	 * 
	 * @param factory pamięci graficzna programu
	 * @param w szerokość przeszukiwanej okolicy
	 * @param h wysokość przeszukiwanej okolicy
	 * @param winw szerokość okna
	 * @param winh wysokość okna
	 */
	PlanetzPicker( const MEM::MISC::GfxPlanetFactory * factory , int w , int h , int winw , int winh );
	/** 
	 * @brief Sprząta po klasie.
	 */
	virtual ~PlanetzPicker();

	/** 
	 * @brief Wyświetla planety do tylnego bufora i sprawdz,
	 * która jest nabliżej widza.
	 * 
	 * @param x pozycja myszki w osi OX
	 * @param y pozycja myszki w osi OY
	 */
	void render( int x , int y );

	/** 
	 * @brief Sprawdza która planeta została wybrana.
	 * 
	 * @return id w buforze wybranej planety.
	 */
	int getId();

	/** 
	 * @brief Reaguje na zmianę wielkości okna.
	 * 
	 * @param w szerokość okna.
	 * @param h wysokość okna.
	 */
	void resize( int w , int h );

	/** 
	 * @brief Wywoływane w reakcji na wciśnięcie klawisza myszy.
	 * Sprawdza która planeta znajduje się pod myszą, i wywołuje
	 * sygnał sigPlanetPicked z jej aktualnym id w buforze.
	 * 
	 * @param b kod przycisku myszy
	 * @param x miejsce przyciśnięcia w osi OX
	 * @param y miejsce przyciśnięcia w osi OY
	 * 
	 * @return zawsze false
	 */
	virtual bool on_button_down( int , int , int );

	/** 
	 * @brief Sygnał informujący o wybraniu planety. Kiedy wybrana
	 * jest pusta przestrzeń, wywoływany z -1 jako id.
	 */
	boost::signal<void (int)> sigPlanetPicked;
private:
	void resizeNames();
	void generate_fb_textures();
	GLuint generate_sphere_texture( int w , int h );

	const MEM::MISC::GfxPlanetFactory * const factory;

	int w , h;
	int winw , winh;

	GFX::Shader vs , gs , fs;
	GFX::Program pr;

	GLuint fboId;

	GLuint depthTex;
	GLuint colorTex;

	GLuint sphereTex;

	GLint sphereTexId;

	GLint outloc;

	uint32_t* buffNames;
	float* buffDepth;

	GLint radiusId , namesId;

	MEM::MISC::BufferGl<uint32_t> names;

	uint32_t max;
};
} // UI

#endif /* __PLANETZ_PICKER_H__ */

