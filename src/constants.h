#ifndef __CONSTANTS_H__

#define __CONSTANTS_H__

#define ROT_BUTTON SDL_BUTTON_RIGHT
#define PICK_BUTTON SDL_BUTTON_LEFT
#define SET_BUTTON SDL_BUTTON_LEFT

#define LEFT_CAM_KEY_0	SDLK_LEFT 
#define RIGHT_CAM_KEY_0	SDLK_RIGHT
#define FWD_CAM_KEY_0	SDLK_UP   
#define BCK_CAM_KEY_0	SDLK_DOWN 
#define UP_CAM_KEY_0	SDLK_SPACE
#define DOWN_CAM_KEY_0	SDLK_LCTRL

#define LEFT_CAM_KEY_1	SDLK_a
#define RIGHT_CAM_KEY_1	SDLK_d
#define FWD_CAM_KEY_1	SDLK_w
#define BCK_CAM_KEY_1	SDLK_s
#define UP_CAM_KEY_1	SDLK_q
#define DOWN_CAM_KEY_1	SDLK_e

#define CAM_ROT_SPEED 1.15
#define BASE_CAM_SPEED 0.15

#define VIEW_LENGHT 1000000
#define VIEW_FOV 75.0

#define PI 3.1415
#define PI2 6.283

#define PATH_MAX_LEN 256
#if defined(linux)
#include <string>
std::string get_bin_path();
static std::string BINARY_PATH = get_bin_path();
#define DATA_PATH (BINARY_PATH + std::string("data/"))
#else
#define DATA_PATH ("data/")
#define BINARY_PATH ("")
#endif
#define BIN(x)   (BINARY_PATH+std::string(x))
#define DATA(x)  (DATA_PATH+std::string(x))
#define SAVES(x) (DATA_PATH+std::string("saves/")+std::string(x))

#endif /* __CONSTANTS_H__ */

