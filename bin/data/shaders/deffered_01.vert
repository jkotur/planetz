#version 130 

varying in float radius;
varying in float model;

varying out float radiuses;
varying out float models;

void main()
{	
	radiuses = radius;
	models   = model;
	gl_Position = gl_ModelViewMatrix * gl_Vertex;
}

