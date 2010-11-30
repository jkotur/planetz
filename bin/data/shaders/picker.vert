#version 130 

in float radius;
in float   name;

out float radiuses;
out float names;

void main()
{
	names    = name;
	radiuses = radius;
	gl_Position = gl_ModelViewMatrix * gl_Vertex;
}

