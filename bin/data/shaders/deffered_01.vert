#version 130 

uniform sampler1D materials;

varying in float radius;
varying in int   model;

varying out float radiuses;
varying out vec4  maters1;
varying out vec4  maters2;

void main()
{	
	radiuses = radius;
	gl_Position = gl_ModelViewMatrix * gl_Vertex;

	maters1 = texelFetch( materials , model   , 0 );
	maters2 = texelFetch( materials , model+1 , 0 );
}

