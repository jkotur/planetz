#version 130

uniform sampler2D sphere;

flat in uint name;

out uvec4 ints;

void main()
{
	float a = texture2D( sphere , gl_TexCoord[0].st ).r;
	ints = uvec4(name,0u,0u,0u);
	if( a <= 0.5 )
		gl_FragDepth = 1.0;
	else	gl_FragDepth = gl_FragCoord.z;
}

