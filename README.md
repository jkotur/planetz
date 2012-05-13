# Planetz

This project was created for BSc diploma. Its main goal was to develop real-time planetary system simulator that can handle over 10 000 planets with good visualisation. This was successfully achieved on computers from year 2010 (geforce 9800GTX+). It is important that all calculations are preformed on GPU and practically no data transfer is required during actual simulation. We used OpenGL and CUDA technologies to make full use of GPU capabilities.

[Cloud]: http://github.com/jkotur/planetz/raw/master/screens/cloud.png "Planets cloud"
[Sun]: http://github.com/jkotur/planetz/raw/master/screens/cloud_sun.png "Planets cloud with one huge sun"
[Duet]: http://github.com/jkotur/planetz/raw/master/screens/duet.png "Two big planets in stable system"
[Lighting 1]: http://github.com/jkotur/planetz/raw/master/screens/lighting_massive.png "Huge plantes system with copule light sources"
[Lighting 2]: http://github.com/jkotur/planetz/raw/master/screens/lighting_one.png "Big plantes system with one light source"
[Trajectory]: http://github.com/jkotur/planetz/raw/master/screens/trajectory.png "Small planetary system with central light and trajectory drawing on"

## Overview

![Cloud][Cloud]
![Sun][Sun]

## Gravity simulation

![Duet][Duet]
![Trajectory][Trajectory]

## Deferred rendering

Rendering system is based on deferred rendering technique which allows to render huge amount of planets with relatively big number of light sources. Planets are rendered as 2D circles on buffer with additional information about depth and texture color. Based on these informations deferred lighting is calculated. Moreover, light sources have finite range, so if sun is far away from planets, no calculations are performed.

![Lighting 1][Lighting 1]
![Lighting 2][Lighting 2]

