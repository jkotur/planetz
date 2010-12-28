#version 130 

in float radius;

out float radiuses;

void main()
{
	radiuses = radius;
	gl_Position = gl_ModelViewMatrix * gl_Vertex + vec4(0,0,-.1,0);
}

