
varying in int model;
varying out bool is_light;

void main()
{
	is_light = ( model == 6 );
	gl_Position = gl_ModelViewMatrix * gl_Vertex;
}

