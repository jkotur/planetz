#ifndef __CAMERA_H__

#define __CAMERA_H__

#include <boost/signal.hpp>

#include "input/driver.h"
#include "util/vector.h"
#include "gfx/drawable.h"

/** 
 * @brief obiekt odpowiedzilny za zachowanie kamery. Powinien on być
 * wyświetlony funkcją draw przed wszystkimi innymi obiektami które mają
 * być widziane z tej kamery.
 */
class Camera : public GFX::Drawable {
public:
	/** 
	 * @brief Konstrukotr.
	 * 
	 * @param pos pozycja kamery
	 * @param lookat punkt na który patrzy kamera
	 * @param up wektor góry kamery
	 */
	Camera ( const Vector3& pos 
		,const Vector3& lookat 
		,const Vector3& up );
	virtual ~Camera();

	/** 
	 * @brief Inicjalizuje kamere.
	 */
	void init();

	/** 
	 * @brief Funkcja reagująca na wciśnięcie klawisza.
	 * 
	 * @param k kod klawisza
	 */
	void on_key_down( int k );
	/** 
	 * @brief Funkcja reagująca na ruch myszki.
	 * 
	 * @param x nowa pozycja myszki w osi OX
	 * @param y nowa pozycja myszki w osi OY
	 */
	void on_mouse_motion( int x , int y );
	/** 
	 * @brief Funkcja reagująca na odciśnięcie guzika myszki.
	 * 
	 * @param b kod guzika
	 * @param x pozycja gdzie nastąpiło odciśnięcie guzika w osi OX
	 * @param y pozycja gdzie nastąpiło odciśnięcie guzika w osi OY
	 */
	void on_button_up( int , int , int );
	/** 
	 * @brief Funkcja reagująca na wciśnięcie guzika myszki.
	 * 
	 * @param b kod guzika
	 * @param x pozycja gdzie nastąpiło wciśnięcie guzika w osi OX
	 * @param y pozycja gdzie nastąpiło wciśnięcie guzika w osi OY
	 * 
	 * @return zawsze zwraca false. Nie blokuje wywołania kolejnych sygnałów.
	 */
	bool on_button_down( int , int , int );
	/** 
	 * @brief Funkcja ustawia widok z kamery w API opengla
	 */
	virtual void draw() const;
	/** 
	 * @brief Funkcja powinna być wywoływana co klatkę animacji.
	 */
	void signal();

	/** 
	 * @brief Ustawia punkt widzenia kamery.
	 * 
	 * @param pos miejsce gdzie stoi kamera
	 * @param lookat punkt na który patrzy kamera
	 * @param up góra kamery
	 */
	void set_perspective(
		 const Vector3& _pos 
		,const Vector3& _lookat 
		,const Vector3& _up )
	{
		pos = _pos;
		lookat = _lookat;
		up = _up;

		init();
	}

	/** 
	 * @brief Ustawia prędkość poruszania się kamery.
	 * 
	 * @param s prędkość
	 */
	void set_speed( double _s )
	{	move_speed = _s; }

	/** 
	 * @brief Zwraca pozycje kamery.
	 */
	Vector3 get_pos() const
	{	return pos; }

	/** 
	 * @brief Zwraca punkt na który patrzy kamera.
	 */
	Vector3 get_lookat() const 
	{	return lookat; }

	/** 
	 * @brief Zwraca góra kamery.
	 */
	Vector3 get_up() const 
	{	return up; }

	/** 
	 * @brief Sygnał wywoływany gdy kamera się obróci.
	 */
	boost::signal<void (float*)> sigAngleChanged;

//        void set_joy( CLocationDriver * _j )
//        {	joy = _j; }
private:
	bool emit_angle_changed_signal();
	
	Vector3 pos , lookat , up;
	Vector3 right , forward;

	int ox , oy; /**< poprzednia pozycja myszy */

	double speed;
	double move_speed;

	bool rot;

//        CLocationDriver * joy;
};


#endif /* __CAMERA_H__ */

