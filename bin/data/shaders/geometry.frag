//CIRL GPU Fragment Program: Derek Anderson and Robert Luke
// very simple fragment shader

void main()
{	
	//Yeah, yeah, yeah ... we just color the pixel
	// this example is showing off geometry shaders, not fragments! 
	//Shade to blue!
	gl_FragColor = gl_Color;
}
