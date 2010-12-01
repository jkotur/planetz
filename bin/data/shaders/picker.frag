
uniform sampler2D sphere;

in float name;

void main()
{
	float a = texture2D( sphere , gl_TexCoord[0].st ).r;
	gl_FragColor.rgba = vec4(name,0,0,a);
}

