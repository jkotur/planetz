#version 130 

uniform sampler1D materialsTex;

in float radius;
in int   model;
in int   texId;
in vec2 atmData;
in vec3 atmColor;

out float radiuses;
out float texIds;

out float atmRadiuses;
out vec3  atmColors;

out vec4  maters1;
out vec4  maters2;

void main()
{	
	radiuses = radius;
	texIds   = texId;
	atmRadiuses = atmData.y;
	atmColors = atmColor;

	gl_Position = gl_ModelViewMatrix * gl_Vertex;

	maters1 = texelFetch( materialsTex , model   , 0 );
	maters2 = texelFetch( materialsTex , model+1 , 0 );
}

