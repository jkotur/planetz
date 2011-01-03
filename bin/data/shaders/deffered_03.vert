#version 130

in vec3 light;
out float em;

in  int model;
out int models;

void main()
{
	em = light.r;
	models = model;
	gl_Position = gl_ModelViewMatrix * gl_Vertex;
}

