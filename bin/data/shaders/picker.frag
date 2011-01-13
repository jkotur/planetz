#version 130

uniform sampler2D sphere;

flat in uint name;

out uvec4 ints;

void main()
{
	float a = texture2D( sphere , gl_TexCoord[0].st ).r;
	ints = uvec4(name,0,0,a);
}

