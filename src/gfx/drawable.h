#ifndef __DRAWABLE_H__

#define __DRAWABLE_H__

#include <cstdlib>

namespace GFX
{
	class Gfx;

	/** 
	 * @brief Bazowa klasa dla wszyskitch elementów które chcą
	 * być wyświetlane przy pomocy klasy GFX::Gfx
	 */
	class Drawable {
	public:
		Drawable () : gfx(NULL) {}
		virtual ~Drawable() {}
		
		/** 
		 * @brief funkcja która jest wywoływana co klatkę graficzną.
		 * Wyświetla obiekt na ekran.
		 */
		virtual void draw() const =0;
		/** 
		 * @brief funkcja przygotowująca klasę do wyśwtlania. Wywoływana jest
		 * po niezbędnej inizjalizacji klasy GFX::Gfx, oraz wstępnej inizjalizaji
		 * GFX::Drawable.
		 */
		virtual void prepare() {}

		/** 
		 * @brief funkcja odpowiedzialna za reakcje renderera na zmianę wielkości
		 * ekranu
		 * 
		 * @param width nowa szerokość ekranu
		 * @param height nowa wysokość ekranu
		 */
		virtual void resize(
				unsigned int width ,
				unsigned int height) {}

		/** 
		 * @brief funkcja reagująca na zmianę konfiguracji klasy GFX::Gfx, 
		 * która może bezpośrednio wpływać na konfigurację wyświetlacza
		 */
		virtual void update_configuration() {}

		/** 
		 * @brief funkcja przy pomocy której GFX::Gfx rejestruje się do
		 * instancji GFX::Drawable
		 * 
		 * @param _g menadżer wyświetlania
		 */
		virtual void setGfx( Gfx* _g )
		{	gfx = _g; prepare(); }
	protected:
		Gfx* gfx;
	};
}

#endif /* __DRAWABLE_H__ */

