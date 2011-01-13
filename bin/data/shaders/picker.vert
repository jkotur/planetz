#version 130 

in float radius;
in uint name;

out float radiuses;
out uint names;

void main()
{
	names    = name;
	radiuses = radius;
	gl_Position = gl_ModelViewMatrix * gl_Vertex;
}

