
uniform sampler2D texture;

void main()
{
	gl_FragColor.rgba = texture2D( texture , gl_TexCoord[0].st );
}

