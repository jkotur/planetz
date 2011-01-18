#version 130 

in float radius;
in vec2 atmData;
in vec3 atmColor;

out vec3 colors;
out float radiuses;

void main()
{
	colors = atmColor;
	radiuses = atmData.y * radius;
	gl_Position = gl_ModelViewMatrix * gl_Vertex + vec4(0,0,-.1*radiuses,0);
}

