#version 130

in vec3 light;
out float em;

in  int model;
out int models;

in  float radius;
out float radiuses;

void main()
{
	em = light.r;
	models = model;
	radiuses = radius;
	gl_Position = gl_ModelViewMatrix * gl_Vertex;
}

