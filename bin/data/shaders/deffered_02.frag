
uniform sampler2D gbuff1;
uniform sampler2D gbuff2;

void main()
{	
	vec4 gdat1 = texture2D( gbuff1 , gl_TexCoord[0].st );
	vec4 gdat2 = texture2D( gbuff2 , gl_TexCoord[0].st );

	if( gdat2.w == 0.0 ) {
		gl_FragColor = gdat1;
	} else {
		gl_FragColor.rgb = vec3(1,1,0) * gdat2.z;
		gl_FragColor.a   = gdat1.a;
	}
}

