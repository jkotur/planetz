#version 130

in float emissive;
out float em;

in  int model;
out int models;

void main()
{
	em = emissive;
	models = model;
	gl_Position = gl_ModelViewMatrix * gl_Vertex;
}

