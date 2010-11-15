
uniform sampler2D sph_pos;

void main()
{	
	vec4 tex = texture2D( sph_pos , gl_TexCoord[0].st );
	gl_FragColor = gl_Color * .75;
	gl_FragColor.g*= tex.z;
        gl_FragColor.a = tex.a;
}

